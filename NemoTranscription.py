import numpy as np
import librosa
from IPython.display import Audio, display
import pyaudio
import plotly
import soundfile as sf
import nemo.collections.asr as nemo_asr
import time


class AudioTranscription:
    def __init__(self, model_name='QuartzNet15x5Base-En', chunkDuration=120, overlapChunkDuration=20,
                 windowWidth=5, fraction=0.2):
        self.model_name = model_name
        self.chunkDuration = chunkDuration
        self.overlapChunkDuration = overlapChunkDuration
        self.windowWidth = windowWidth
        self.fraction = fraction
        self.asr_mode = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name)

    def updateWordList(self, arrayOfSamplesToBeUpdated, UpperUpdate, LowerUpdate, UpperUpdateIndex):
        arrayOfSamplesToBeUpdated[UpperUpdateIndex] = UpperUpdate
        arrayOfSamplesToBeUpdated[UpperUpdateIndex + 1] = LowerUpdate

    def windowSearch(self, upperWordList, lowerWordList):

        global tempFirstIndex

        finalUpper = upperWordList
        FinalLower = lowerWordList

        # get lengths of list
        upperLength = len(upperWordList)
        lowerLength = len(lowerWordList)

        # get sliding length of the upper and lower chunk of word
        upperSlideLength = upperLength - int(upperLength * self.fraction)
        lowerSlideLength = int(lowerLength * self.fraction)

        # the overlapped chunk list of words
        upperNewList = upperWordList[upperSlideLength:]
        lowerNewList = lowerWordList[:lowerSlideLength]

        firstIndex = 0
        secondIndex = 0
        arrayOfWords = []
        stopSearching = False
        lengthOfUpperLimit = len(upperNewList) + 1 - self.windowWidth
        lengthOfLowerLimit = len(lowerNewList) + 1 - self.windowWidth
        for upperLimitIndex in range(0, lengthOfUpperLimit):
            arrayOfWords.clear()
            if stopSearching:
                break
            for lowerLimitIndex in range(0, lengthOfLowerLimit):
                countOfEqualWords = 0
                for i in range(0, self.windowWidth):
                    if upperNewList[upperLimitIndex + i] == lowerNewList[lowerLimitIndex + i]:
                        if i == 0:
                            tempFirstIndex = upperLimitIndex + i
                        arrayOfWords.append(upperNewList[upperLimitIndex + i])
                        countOfEqualWords += 1
                        if countOfEqualWords == self.windowWidth:
                            secondIndex = lowerLimitIndex + i
                            firstIndex = tempFirstIndex
                            stopSearching = True
                            break
        if stopSearching:
            finalUpper = upperWordList[:(upperSlideLength + firstIndex + self.windowWidth)]
            FinalLower = lowerWordList[secondIndex + 1:]

        return finalUpper, FinalLower

    def transcribeAudio(self, audiofile=""):

        transcribedWord = ""

        # extracting audio properties
        signal, sample_rate = librosa.load(audiofile, sr=None)
        audioSignalLength = len(signal)
        duration = audioSignalLength / sample_rate
        print("audio duration: ", duration, "seconds")
        print("sample_rate: ", sample_rate)

        # check audio length
        if duration > self.chunkDuration:

            # chunk initialization
            chunkSize = sample_rate * self.chunkDuration  # a minute sample chunk of data
            chunk_offset = sample_rate * self.overlapChunkDuration  # overllaping chunk files
            samplesOfChunks = audioSignalLength // chunkSize
            totalNumberOfChunkSample = (samplesOfChunks * chunkSize)
            reminderOfSamples = audioSignalLength - totalNumberOfChunkSample
            samplesToAdded = chunkSize - reminderOfSamples

            if samplesToAdded != 0:
                # convert to list
                signal = signal.tolist()
                # newSignal = signal + ([0] * samplesToAdded)
                # signal = newSignal
                samplesOfChunks = samplesOfChunks + 1

            print("number of chunks: ", samplesOfChunks)

            # store all chunks in array of list
            arrayOfSamples = []
            for i in range(samplesOfChunks):
                upperL = (i + 1) * chunkSize
                lowerL = i * chunkSize
                audioName = "newSample.wav"
                if i != 0:
                    lowerL = lowerL - chunk_offset
                if i == (samplesOfChunks - 1):
                    newList = signal[lowerL:]
                else:
                    newList = signal[lowerL:upperL]

                # save audio
                sf.write(audioName, newList, sample_rate, subtype='PCM_24')

                # transcribe Audio
                files = [audioName]
                text = self.asr_mode.transcribe(paths2audio_files=files)[0]
                text = text.strip()

                # check if the audio has transcribed word
                if len(text) > 0:
                    # store split transcribe audio in array of list
                    arrayOfSamples.append(text.split())
                    # display(Audio(data=newList, rate=sample_rate))

            numberOfIteration = len(arrayOfSamples) - 1
            for wordsIndex in range(numberOfIteration):
                # window search similar words and remove them
                upperWord, lowerWord = self.windowSearch(arrayOfSamples[wordsIndex], arrayOfSamples[wordsIndex + 1])
                # update the clean list of words into list of words
                self.updateWordList(arrayOfSamples, upperWord, lowerWord, wordsIndex)
                # concat all split words into a sentence
                for listOfWords in upperWord:
                    transcribedWord = transcribedWord + " " + listOfWords
                if wordsIndex == numberOfIteration - 1:
                    for listOfWords in lowerWord:
                        transcribedWord = transcribedWord + " " + listOfWords

        else:
            file = [audiofile]
            transcribedWord = self.asr_mode.transcribe(paths2audio_files=file)[0]

        return transcribedWord


"""if __name__ == "__main__":
    tr = AudioTranscription()
    text = tr.transcribeAudio("newSample3.wav")

    print("text: \n", text)"""
