"""HercUNet nnU-Net trainer classes (the run301 / v0 chain).

FocalTversky → …Sep → …SepSkel → …SepSkelSym → …_Iter → …_IterDagger → AffinityMalis_IterDagger.

Each is a thin subclass of the previous (or of stock ``nnUNetTrainer``); all loss / prev-channel /
affinity / MALIS logic lives in the sibling library modules (:mod:`hercunet.training.losses`,
``prev_channel``, ``prev_transform``, ``affinity``, ``malis``). Unlike the research repo these are
**not** copied into the ``nnunetv2`` package — :func:`hercunet.train.launch.fit` imports the class and
instantiates it directly (no ``-tr`` string discovery). Class names are kept identical to the research
pipeline so existing checkpoints' ``trainer_name`` still resolves.
"""
