import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from config import DATA_PATHS


def get_description(web: BeautifulSoup) -> str:
    """
    Returns the description of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: description (string)
    """
    descriptions = web.findAll('div', attrs={'class': 'description'})
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('p')
        for j, p in enumerate(paragraphs):
            if i == 2 and j == 1:
                return p.text


def get_code(web: BeautifulSoup) -> str:
    """
    Returns the code of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: code (string)
    """
    descriptions = web.findAll('div', attrs={'class': 'code'})
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('p')
        return paragraphs[1].text


def get_atr_label(web: BeautifulSoup) -> list:
    """
    Returns the alternative label of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: alternative label (list of strings)
    """
    descriptions = web.findAll('div', attrs={'class': 'alternative-labels'})
    alter_labels = []
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('p')
        for j, p in enumerate(paragraphs):
            alter_labels.append(p.text)
    return alter_labels


def get_skills(web: BeautifulSoup) -> list:
    """
    Returns the skills of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: alternative label (list of string)
    """
    descriptions = web.findAll('div', attrs={'class': 'essential-skills-list'})
    skills = []
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('a')
        for j, p in enumerate(paragraphs):
            skills.append(p.text)
    return skills


def get_knowledge(web: BeautifulSoup) -> list:
    """
    Returns the Essential Knowledge the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: alternative label (list of string)
    """
    descriptions = web.findAll('div', attrs={'class': 'essential-knowledge-list'})
    knowledge = []
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('a')
        for j, p in enumerate(paragraphs):
            knowledge.append(p.text)
    return knowledge


def get_opt_skills(web: BeautifulSoup) -> list:
    """
    Returns the Optional Skills and Competences of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: alternative label (list of string)
    """
    descriptions = web.findAll('div', attrs={'class': 'optional-skills-list'})
    opt_skills = []
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('a')
        for j, p in enumerate(paragraphs):
            opt_skills.append(p.text)
    return opt_skills


def get_opt_knowledge(web: BeautifulSoup) -> list:
    """
    Returns the Optional Skills and Competences of the Job Title.
    :param web: BeautifulSoup object of the website.
    :return: alternative label (list of string)
    """
    descriptions = web.findAll('div', attrs={'class': 'optional-knowledge-list'})
    opt_knowledge = []
    for i, d in enumerate(descriptions):
        paragraphs = d.findAll('a')
        for j, p in enumerate(paragraphs):
            opt_knowledge.append(p.text)
    return opt_knowledge


if __name__ == "__main__":
    columns = ['occupation', 'code', 'description', 'alternative_labels', 'skills',
               'knowledge', 'opt_skills', 'opt_knowledge']

    for phase in DATA_PATHS:
        metadata_name = DATA_PATHS[phase]['metadata']
        data_name = DATA_PATHS[phase]['data']

        a = []
        meta_data = pd.read_csv(f'../data/{metadata_name}')

        for i, url in enumerate(tqdm(meta_data['conceptUri'], desc='Progress Bar')):
            redirected_url = 'https://esco.ec.europa.eu/en/classification/occupation?uri='
            r = requests.get(redirected_url + url)
            soup = BeautifulSoup(r.content, 'html5lib')
            try:
                occupation = str(soup.findAll('h3')[0].string).strip()
                code = get_code(soup)
                description = get_description(soup)
                alternative_labels = get_atr_label(soup)
                skills = get_skills(soup)
                knowledge = get_knowledge(soup)
                opt_skills = get_opt_skills(soup)
                opt_knowledge = get_opt_knowledge(soup)
                row = [occupation, code, description, alternative_labels, skills, knowledge, opt_skills, opt_knowledge]
                a.append(row)
            except:
                print('Skipping-{} '.format(url))

        print(f'{phase} is done.')

        df = pd.DataFrame(a, columns=columns)
        df.to_excel(f"../data/{data_name}")
