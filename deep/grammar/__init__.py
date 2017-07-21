
from .abstract import AbstractGrammar
from .simple import SimpleGrammar
from .thingtalk import ThingtalkGrammar

def create_grammar(grammar_type, input_file):
    if grammar_type == "tt":
        return ThingtalkGrammar(input_file)
    elif grammar_type == "simple":
        return SimpleGrammar(input_file)
    else:
        raise ValueError("Invalid grammar %s" % (grammar_type,))