from NemoTranscription import AudioTranscription
import yaml
from yaml.loader import SafeLoader
import logging
from Sum_Script import t5inference


def load_configs(filename):
    with open(filename) as f:
        data = yaml.load(f,Loader=SafeLoader)

    return data

def get_logger(logger_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    stream_h = logging.StreamHandler()
    file_h = logging.FileHandler(logger_file) # logger files keeps tracks of all your logging messages

    format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_h.setFormatter(format)
    file_h.setFormatter(format)

    logger.addHandler(stream_h)
    logger.addHandler(file_h)

    return logger




def transcribe_and_summarize(wav_path):

    logger = get_logger('logs.log')
    config = load_configs('config.yaml')

    tr = AudioTranscription()
    t5 = t5inference(config,logger)
    text = tr.transcribeAudio(wav_path)
    prefix = 'summarize: '
    output = t5.infer_single(text,prefix)

    return text,output


    
if __name__ == "__main__":
    text, output = transcribe_and_summarize('newSample2.wav')
    print("text: ",text)
    print("summary: ",output)
