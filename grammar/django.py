'''
Created on Apr 24, 2018

@author: gcampagn
'''

from .shift_reduce_grammar import ShiftReduceGrammar

from collections import OrderedDict
from keyword import kwlist


class DjangoGrammar(ShiftReduceGrammar):
    '''
    The grammar of Django
    '''

    def __init__(self, filename=None, **kw):
        super().__init__(**kw)
        self.entities = []

        '''
        Python2.7 Grammar
        See this link: https://docs.python.org/2/reference/grammar.html
        '''

        GRAMMAR = OrderedDict({
            '$input': [
                # simple stmt
                ('$expr_stmt',),
                ('$print_stmt',),
                ('$del_stmt',),
                ('$pass_stmt',),
                ('$flow_stmt',),
                ('$import_stmt',),
                ('$global_stmt',),
                ('$exec_stmt',),
                ('$assert_stmt',),

                # flow stmt
                ('$break_stmt',),
                ('$continue_stmt',),
                ('$return_stmt',),
                ('$raise_stmt',),
                ('$except_clause',),
                ('$yield_stmt',),

                # compount stmt
                ('$if_stmt',),
                ('$elif_stmt',),
                ('$else_stmt',),
                ('$while_stmt',),
                ('$for_stmt',),
                ('$try_stmt',),
                ('$with_stmt',),
                ('$funcdef',),
                ('$classdef',),
                ('$decorated',)],

            '$decorator': [('@', '$dotted_name', '<<EOF>>'),
                           ('@', '$dotted_name', '(', '$arglist', ')', '<<EOF>>')], #fixme
            '$decorators': [('$decorator',),
                            ('$decorators', '$decorator')],
            '$decorated': [('$decorators', '$classdef'),
                           ('$decorators', '$funcdef')],
            '$funcdef': [('def', '$ident', '$parameters', ':')],
            '$parameters': [('(', '$varargslist', ')'),
                            ('(', ')')],

            #########################################

            '$fpvar': [('$fpdef',),
                       ('$fpdef', '=', '$test')],
            '$varargslist_tmp': [('$fpvar', ','),
                                 ('$varargslist_tmp', '$fpvar', ',')],
            '$varargslist': [('$fpvar',),
                             ('$varargslist_tmp',),
                             ('$varargslist_tmp', '$fpvar'),
                             ('$varargslist_tmp', '*', '$ident'),
                             ('$varargslist_tmp', '*', '$ident', ',', '**', '$ident'),
                             ('$varargslist_tmp', '**', '$ident')],
            ##########################################

            '$fpdef': [('$ident',),
                       ('(', '$fplist', ')')],
            '$fplist': [('$fpdef', ','),
                        ('$fpdef',),
                        ('$fpdef', ',', '$fpdef'),
                        ('$fpdef', ',', '$fpdef', ',')],

            '$expr_stmt': [('$testlist', '$augassign', '$yield_expr'),
                           ('$testlist', '$augassign', '$testlist'),
                           ('$testlist',),
                           ('$testlist', '=', '$yield_expr'),
                           ('$testlist', '=', '$testlist')],

            '$augassign': [('+=',),
                           ('-=',),
                           ('*=',),
                           ('/=',),
                           ('%=',),
                           ('&=',),
                           ('|=',),
                           ('^=',),
                           ('<<=',),
                           ('>>=',),
                           ('**=',),
                           ('//=',)],

            '$print_stmt_tmp_2': [('$test',),
                                  ('$print_stmt_tmp_2', ',', '$test')],
            '$print_stmt_tmp': [('$print_stmt_tmp_2',),
                               ('$print_stmt_tmp_2', ',')],
            '$print_stmt': [('print',),
                            ('print', '$print_stmt_tmp')], # fixme
            '$del_stmt': [('del', '$exprlist')],
            '$pass_stmt': [('pass',)],
            '$flow_stmt': [],
            '$break_stmt': [('break',)],
            '$continue_stmt': [('continue',)],
            '$return_stmt': [('return',),
                             ('return', '$testlist')],
                             #('return', '$ident', '.', '$ident')],
                             #('return', '$dotted_name')], #hack
            '$yield_stmt': [('$yield_expr',)],
            '$raise_stmt': [('raise',),
                            ('raise', '$test'),
                            ('raise', '$test', ',', '$test'),
                            ('raise', '$test', ',', '$test', ',', '$test')],
            '$import_stmt': [('$import_name',),
                             ('$import_from',)],
            '$import_name': [('import', '$dotted_as_names')],
            ########################
            '$import_from': [('from', '$dotted_name'),
                             ('from', '.', '$dotted_name'),
                             ('from', '$dotted_name', 'import', '*'),
                             ('from', '$dotted_name', 'import', '(', '$import_as_names', ')'),
                             ('from', '$dotted_name', 'import', '$import_as_name'),
                             ('from', '$dotted_name', 'import', '$import_as_name', ',', '$import_as_name')], #hack
            '$import_as_name': [('$ident',),
                                ('$ident', 'as', '$ident')],
            '$dotted_as_name': [('$dotted_name',),
                                ('$dotted_name', 'as', '$ident')],
            '$import_as_names_tmp': [('$import_as_name',),
                                     ('$import_as_names_tmp', ',', '$import_as_name')],
            '$import_as_names': [('$import_as_names_tmp',),
                                 ('$import_as_names_tmp', ',')],
            '$dotted_as_names': [('$dotted_as_name',),
                                 ('$dotted_as_names', ',', '$dotted_as_name')],
            '$dotted_name': [('$ident',),
                             ('$dotted_name', '.', '$ident')],
            '$global_stmt': [('global', '$ident'),
                             ('$global_stmt', ',', '$ident')],
            '$exec_stmt': [('exec', '$expr'),
                           ('exec', '$expr', 'in', '$test')],
            '$assert_stmt': [('assert', '$test'),
                             ('assert', '$test', ',', '$test')],

            '$if_stmt': [('if', '$test', ':',)],
            '$elif_stmt': [('elif', '$test', ':')],
            '$else_stmt': [('else', ':')],
            '$while_stmt': [('while', '$test', ':',)],
            '$for_stmt': [('for', '$exprlist', 'in', '$testlist', ':',)],
            '$try_stmt': [('try', ':')],
            '$with_stmt': [('with', '$with_item', ':',)],
            '$with_item': [('$test',),
                           ('$test', 'as', '$expr')],

            # '$except_clause_tmp': [('$test',),
            #                        ('$test', 'as', '$test'),
            #                        ('$test', ',', '$test')],
            '$except_clause': [('except',),
                               ('except', '$ident', ':')], #Hack
                               #('except', '$except_clause_tmp')],

            '$testlist_safe': [('$old_test',)],
            '$old_test': [('$or_test',),
                          ('$old_lambdef',)],
            '$old_lambdef': [('lambda', ':', '$old_test'),
                             ('lambda', '$varargslist', ':', '$old_test')],
            '$test': [('$lambdef',),
                      ('$or_test',),
                      ('$or_test', 'if', '$or_test', 'else', '$test')],
            '$or_test': [('$and_test',),
                         ('$or_test', 'or', '$and_test')],
            '$and_test': [('$not_test',),
                          ('$and_test', 'and', '$not_test')],
            '$not_test': [('not', '$not_test'),
                          ('$comparison',)],
            '$comparison': [('$expr',),
                            ('$comparison', '$comp_op', '$expr')],
            '$comp_op': [('<',),
                         ('>',),
                         ('==',),
                         ('>=',),
                         ('<=',),
                         ('<>',),
                         ('!=',),
                         ('in',),
                         ('not', 'in'),
                         ('is',),
                         ('is', 'not')],
            '$expr': [('$xor_expr',),
                      ('$expr', '|', '$xor_expr')],
            '$xor_expr': [('$and_expr',),
                          ('$xor_expr', '^', '$and_expr')],
            '$and_expr': [('$shift_expr',),
                          ('$and_expr', '&', '$shift_expr')],
            '$shift_expr': [('$arith_expr',),
                            ('$shift_expr', '<<', '$arith_expr'),
                            ('$shift_expr', '>>', '$arith_expr')],  # fixme
            '$arith_expr': [('$term',),
                            ('$arith_expr', '+', '$term'),
                            ('$arith_expr', '-', '$term')],
            '$term': [('$factor',),
                      ('$term', '*', '$factor'),
                      ('$term', '/', '$factor'),
                      ('$term', '%', '$factor'),
                      ('$term', '//', '$factor')],
            '$factor': [('+', '$factor'),
                        ('-', '$factor'),
                        ('~', '$factor'),
                        ('$power',)],
            '$power_tmp': [('$atom',),
                           ('$atom', '$trailer_list')],
            '$trailer_list': [('$trailer',),
                              ('$trailer_list', '$trailer')],
            '$power': [('$power_tmp',),
                       ('$power_tmp', '**', '$factor')],
            '$atom': [('(', '$yield_expr', ')'),
                      ('(', '$testlist_comp', ')'),
                      ('[', '$listmaker', ']'),
                      ('{', '$dictorsetmaker', '}'),
                      ('`', '$testlist1', '`'),
                      ('$ident',),
                      ('NUMBER',),
                      ('$string_list',),
                      ('(', ')'),
                      ('[', ']'),
                      ('{', '}')],
            '$string_list': [('STRING',),
                             ('$string_list', 'STRING')],
            '$listmaker_tmp': [('$test',),
                               ('$listmaker_tmp', ',', '$test')],
            '$listmaker': [('$test', '$list_for'),
                           ('$listmaker_tmp',),
                           ('$listmaker_tmp', ',')],
            '$testlist_comp': [('$test', '$comp_for'),
                               ('$test', ',', '$test')],
            '$lambdef': [('lambda', ':', '$test'),
                         ('lambda', '$varargslist', ':', '$test')],
            '$trailer': [('.', '$ident'),
                         ('(', '$arglist', ')'),
                         ('(', ')'),
                         ('[', '$subscriptlist', ']'),
                         ('[', ']')],
            '$subscriptlist_tmp': [('$subscript',),
                                   ('$subscriptlist_tmp', ',', '$subscript')],
            '$subscriptlist': [('$subscriptlist_tmp',),
                               ('$subscriptlist_tmp', ',')],
            '$subscript': [('.', '.', '.'),
                           ('$test',),
                           (':',),
                           ('$test', ':'),
                           ('$test', ':', '$test'),
                           ('$test', ':', '$sliceop'),
                           ('$test', ':', '$test', '$sliceop'),
                           (':', '$test'),
                           (':', '$test', '$sliceop'),
                           (':', '$sliceop')],
            '$sliceop': [(':',),
                         (':', '$test')],

            '$exprlist_tmp': [('$expr',),
                              ('$exprlist_tmp', ',', '$expr')],
            '$exprlist': [('$exprlist_tmp',),
                          ('$exprlist_tmp', ',')],

            '$testlist_tmp': [('$test',),
                              ('$testlist_tmp', ',', '$test')],
            '$testlist': [('$testlist_tmp',),
                          ('$testlist_tmp', ',')],
            '$dictorsetmaker': [],

            '$classdef': [('class', '$ident', ':',),
                          ('class', '$ident', '(', '$testlist', ')', ':',)],
            '$arglist_tmp': [('$argument', ','),
                             ('$arglist_tmp', '$argument', ',')],
            '$arglist': [('$argument',),
                         ('$argument', ','),
                         ('*', '$test'),
                         ('**', '$test'),
                         ('$arglist_tmp', '$argument'),
                         ('$arglist_tmp', '$argument', ','),
                         ('$arglist_tmp', '*', '$test'),
                         ('$arglist_tmp', '**', '$test')],
            '$argument': [('$test',),
                          ('$test', '$comp_for'),
                          ('$test', '=', '$test')],
            '$list_iter': [('$list_for',),
                           ('$list_if',)],
            '$list_for': [('for', '$exprlist', 'in', '$testlist_safe'),
                          ('for', '$exprlist', 'in', '$testlist_safe', '$list_iter')],
            '$list_if': [('if', '$old_test'),
                         ('if', '$old_test', '$list_iter')],

            '$comp_iter': [('$comp_for',),
                           ('$comp_if',)],
            '$comp_for': [('for', '$exprlist', 'in', '$or_test'),
                          ('for', '$exprlist', 'in', '$or_test', '$comp_iter')],
            '$comp_if': [('if', '$old_test'),
                         ('if', '$old_test', '$comp_iter')],

            '$testlist1': [('$test',),
                           ('$testlist1', ',', '$test')],
            '$yield_expr': [('yield',),
                            ('yield', '$testlist')],

            '$ident': [('IDENT',)]
        })


        self.allfunctions = set()

        idents = set()

        # add things that are keywords in python3 but identifiers in python2
        idents.add('None')
        idents.add('True')
        idents.add('False')
        idents.add('nonlocal')

        #HACK
        #idents.add('Exception')

        with open(filename, 'r') as fp:
            for line in fp:
                for token in line.strip().split(' '):
                    if token:
                        if (token[0].isalpha() or token.startswith('_')) and not token in kwlist:
                            if token.find('$') == -1 and token.find('STR') == -1:
                                idents.add(token)

        idents = list(idents)
        idents.sort()
        self.num_functions = 0
        self.tokens += self.construct_parser(grammar=GRAMMAR,
                                             extensible_terminals={
                                                                   },
                                             copy_terminals={'IDENT': idents,
                                                             'NUMBER': list(map(str, range(200))),
                                                             'STRING': ['STR' + str(i) for i in range(156)]
                                                             })

        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i


