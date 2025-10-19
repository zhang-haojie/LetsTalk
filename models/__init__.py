from .model_cat import VDTcat_models
from .model_cat_long import VDTcatlong_models
from .lite_cat_long import Litecatlong_models


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def get_models(args):
    if 'Litecatlong' in args.model:
        return Litecatlong_models[args.model](
                input_size=args.latent_size,
                context_dim=args.audio_dim,
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                num_frames=args.clip_frames,
                initial_frames=args.initial_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                temp_comp_rate=args.temp_comp_rate,
                gradient_checkpointing=args.gradient_checkpointing
            )
    elif 'VDTcatlong' in args.model:
        return VDTcatlong_models[args.model](
                input_size=args.latent_size,
                context_dim=args.audio_dim,
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                num_frames=args.clip_frames,
                initial_frames=args.initial_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                temp_comp_rate=args.temp_comp_rate,
                gradient_checkpointing=args.gradient_checkpointing
            )
    elif 'VDTcat' in args.model:
        return VDTcat_models[args.model](
                input_size=args.latent_size,
                context_dim=args.audio_dim,
                in_channels=args.in_channels,
                num_classes=args.num_classes,
                num_frames=args.clip_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                temp_comp_rate=args.temp_comp_rate,
                gradient_checkpointing=args.gradient_checkpointing
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
    