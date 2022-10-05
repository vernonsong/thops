# @Time : 2022/9/29 09:41
# @Author : songweinan
# @Software: PyCharm
# 自能成羽翼，何必仰云梯。
# ==============================================================================
import datetime
import warnings

from mmengine.registry import DefaultScope
from mmocr.utils import register_all_modules as register_all_modules_


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmocr into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmocr default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `mmocr`, and all registries will build modules from mmocr's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    # import thops.apis  # noqa: F401,F403
    # import mmocr.datasets  # noqa: F401,F403
    # import mmocr.engine  # noqa: F401,F403
    # import mmocr.evaluation  # noqa: F401,F403
    import thops.models  # noqa: F401,F403

    # import mmocr.structures  # noqa: F401,F403
    # import mmocr.visualization  # noqa: F401,F403
    register_all_modules_(init_default_scope=False)
    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('thops')
        if never_created:
            DefaultScope.get_instance('thops', scope_name='thops')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'thops':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmocr", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmocr". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'thops-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='thops')
