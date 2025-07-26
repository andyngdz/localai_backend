import logging


class LoggerService:
	def init(self):
		logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')


logger_service = LoggerService()
