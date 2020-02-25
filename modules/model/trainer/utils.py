import importlib
import logging


logger = logging.getLogger(__name__)

apex = importlib.util.find_spec('apex')
if apex is not None:
    apex = importlib.import_module('apex')
    logger.warning('Apex was loaded.')
else:
    logger.warning('Apex module was not found.')

