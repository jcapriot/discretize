import os
import os.path
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('discretize', parent_package, top_path)

    config.add_subpackage('utils')
    config.add_subpackage('mixins')
    config.add_subpackage('base')
    config.add_subpackage('pydantic')
    config.add_subpackage('_extensions')

    return config
