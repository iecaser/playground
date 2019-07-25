from absl import app, flags
from loguru import logger

flags.DEFINE_string('a', 'a', 'string a')
flags.DEFINE_string('b', 'b', 'string b')
FLAGS = flags.FLAGS


def main(_):
    # flags.declare_key_flag('a')
    a = FLAGS.a
    b = FLAGS.b
    # flags.declare_key_flag('b')
    logger.info(f'run main: {a} {b}')


if __name__ == '__main__':
    app.run(main)
