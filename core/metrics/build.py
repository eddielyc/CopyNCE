
__metric_factory__ = {}

from loguru import logger


def register_metric(metric_name=""):
    def wrapper(cls):
        if metric_name in __metric_factory__:
            logger.warning(
                f'Overwriting {metric_name} in registry with {cls.__name__}. This is because the name being '
                'registered conflicts with an existing name. Please check if this is not expected.'
            )
        __metric_factory__[metric_name] = cls
        return cls

    return wrapper


def build_metric(metric_type, dump_dir, epoch=None):
    return __metric_factory__[metric_type](dump_dir, epoch)
