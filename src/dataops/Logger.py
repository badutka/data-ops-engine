# import coloredlogs, logging
#
# logger = None
#
#
# def setup_logger():
#     formatter = logging.Formatter("%(message)s")
#     logger = logging.getLogger("dataops-logger")
#
#     logger.setLevel(logging.DEBUG)
#
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)
#
#     fh = logging.FileHandler("logs.log")
#     fh.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
#     # By default the install() function installs a handler on the root logger,
#     # this means that log messages from your code and log messages from the
#     # libraries that you use will all show up on the terminal.
#     # coloredlogs.install(level='DEBUG')
#
#     # If you don't want to see log messages from libraries, you can pass a
#     # specific logger object to the install() function. In this case only log
#     # messages originating from that logger will show up on the terminal.
#     coloredlogs.install(
#         level='DEBUG',
#         logger=logger,
#         level_styles={
#             'debug': {'color': 'blue'},
#             'info': {'color': 'green'},
#             'warning': {'color': 'yellow'},
#             'error': {'color': 'red'},
#             'critical': {'color': 'red', 'bold': True}
#         })
#
#     return logger
#
#
# if logger is None:
#     logger = setup_logger()
