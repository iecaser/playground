from absl import app, flags, logging
from loguru import logger

FLAGS = flags.FLAGS
flags.DEFINE_string(name='name',
                    default='world',
                    help='Input your name.')


def main(_):
    logger.info(f'hello {FLAGS.name}')
    logging.info(f'hello {FLAGS.name}')


app.run(main)
