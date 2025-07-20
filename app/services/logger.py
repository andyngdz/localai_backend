import logging


class LoggerService:
	def start(self):
		logging.basicConfig(
			level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
		)


logger_service = LoggerService()
