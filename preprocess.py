import sys
import numpy as np
import os
import re


def download_glove(glove):
    if os.path.exists(glove):
        return

    print('Downloading glove...')
    with tempfile.TemporaryFile() as tmp:
        with urllib.request.urlopen('http://nlp.stanford.edu/data/glove.42B.300d.zip') as res:
            shutil.copyfileobj(res, tmp)
        with zipfile.ZipFile(tmp, 'r') as glove_zip:
            glove_zip.extract('glove.42B.300d.txt', path=os.path.dirname(glove))
    print('Done')


def add_words(input_words, canonical):
    if isinstance(canonical, str):
        sequence = canonical.split(' ')
    else:
        sequence = canonical
    for word in sequence:
        if word:
            input_words.add(word)

        # if not word:
        #     raise ValueError('Invalid canonical "%s"' % (canonical,))

def save_dictionary(input_words, workdir):
    with open(os.path.join(workdir, 'input_words.txt'), 'w') as fp:
        for word in sorted(input_words):
            print(word, file=fp)

def create_dictionary(input_words, dataset):
    for filename in os.listdir(dataset):
        if not filename != 'train.txt' or filename != 'test.txt':
            continue

        with open(os.path.join(dataset, filename), 'r') as fp:
            for line in fp:
                sentence = line.strip().split('\t')[1]
                add_words(input_words, sentence)
    # FIXME no extra word
    if len(sys.argv) > 4:
        extra_word_file = sys.argv[4]
        print('Adding extra dictionary from', extra_word_file)
        with open(extra_word_file, 'r') as fp:
            for line in fp:
                input_words.add(line.strip())


def trim_embeddings(input_words, workdir, embed_size, glove):


    blank = re.compile('^_+$')

    output_embedding_file = os.path.join(workdir, 'embeddings-' + str(embed_size) + '.txt')
    with open(output_embedding_file, 'w') as outfp:
        with open(glove, 'r') as fp:
            for line in fp:
                stripped = line.strip()
                sp = stripped.split()
                # if sp[0] in HACK:
                #     HACK[sp[0]] = sp[1:]
                if sp[0] in input_words:
                    print(stripped, file=outfp)
                    input_words.remove(sp[0])

        for word in input_words:
            if not word or re.match('\s+', word):
                raise ValueError('Invalid word "%s"' % (word,))
            vector = None
            # if blank.match(word):
            #     # normalize blanks
            #     vector = HACK['____']
            # elif word.endswith('s') and word[:-1] in HACK:
            #     vector = HACK[word[:-1]]
            # elif (word.endswith('ing') or word.endswith('api')) and word[:-3] in HACK:
            #     vector = HACK[word[:-3]]
            # elif word in HACK_REPLACEMENT:
            #     vector = HACK[HACK_REPLACEMENT[word]]
            # elif '-' in word:
            #     vector = np.zeros(shape=(len(HACK['____']),), dtype=np.float64)
            #     for w in word.split('-'):
            #         if w in HACK:
            #             vector += np.array(HACK[w], dtype=np.float64)
            #         else:
            #             vector = None
            #             break
            if vector is not None:
                print(word, *vector, file=outfp)

            else:
                if not word[0].isupper():
                    print("WARNING: missing word", word)
                print(word, *np.random.normal(0, 0.9, (embed_size,)), file=outfp)


def main():
    np.random.seed(1234)

    if len(sys.argv) != 4:
        print("** Usage: python3 " + "/path/to/preprocess.py" + "current working dir" + "Django dataset dir" + "GloVe embeddings dir **")
        sys.exit(1)
    else:
        workdir, dataset, glove = sys.argv[1:4]

    embed_size = 300
    glove = os.path.join(glove, 'glove.42B.300d.txt')
    download_glove(glove)

    input_words = set()
    # add the canonical words for the builtin functions
    # and a few canonical words that are useful
    add_words(input_words, 'function return argument if else call import raise args model params def')

    create_dictionary(input_words, dataset)
    #get_thingpedia(input_words, workdir, snapshot)
    save_dictionary(input_words, workdir)
    trim_embeddings(input_words, workdir, embed_size, glove)


if __name__ == '__main__':
    main()


