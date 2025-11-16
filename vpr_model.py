import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    We use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 backbone_config={},

                 # ---- Aggregator
                 agg_arch='ConvAP',
                 agg_config={},

                 # ---- Train hyperparameters
                 lr=0.03,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 lr_sched='linear',
                 lr_sched_args={
                     'start_factor': 1,
                     'end_factor': 0.2,
                     'total_iters': 4000,
                 },

                 # ----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # Write hyperparameters into a file

        # Dynamically load loss and miner functions
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []  # We will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # Get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []

    # The forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    # Configure the optimizer
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        # Learning rate scheduler
        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'],
                                                 gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Warm up learning rate
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()

    # Loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # We mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # Calculate the % of trivial pairs/triplets
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:  # No online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):  # Some losses might return a tuple (loss, batch_accuracy)
                loss, batch_acc = loss

        # Log batch accuracy
        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # Training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        BS, N, ch, h, w = places.shape

        # Reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images)

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')

        loss = self.loss_function(descriptors, labels)  # Call the loss function

        self.log('loss', loss.item(), logger=True, prog_bar=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # Reset batch accuracy at the end of each epoch
        self.batch_acc = []

    # Validation step
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        # Reset the outputs list at the start of each validation epoch
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]

    def on_validation_epoch_end(self):
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # If we have only one validation set, put the outputs in a list
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            if 'pitts' in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[: num_references]
            q_list = feats[num_references:]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # Reset the outputs list
        self.val_outputs = []
