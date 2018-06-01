import re


def process_elements_query(query):
    j = 0

    for i in ["first", "second", "third", "fourth", "fifth", "sixth"]:
        query.replace(i + " " + "element", str(j) + " " + "element")
        j += 1

    all_matches = re.findall(r"(\d)+\selements", query)
    for match in all_matches:
        query.replace(match + " elements", str(int(match)-1) + " elements")

    return query


def clean_code_of_empty_strings_with2spaces(code_file):
    f_code_cleaned = open('./raw_data/all.code.cleaned', 'w')

    for idx, code in enumerate(open(code_file)):
        code = code.strip()
        code = re.sub(r'(?:\'\s{2}\'|\"\s{2}\")', r'', code)
        f_code_cleaned.write(code + '\n')
    f_code_cleaned.close()




if __name__ == '__main__':
    code_file = '/Users/Mehrad/Documents/GitHub/Read_Django/raw_data/all.code'
    clean_code_of_empty_strings_with2spaces(code_file)
