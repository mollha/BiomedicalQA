from pathlib import Path
import xml.etree.ElementTree as et
from tqdm import tqdm
import requests
import json

base_path = Path(__file__).parent
env_path = (base_path / '../../../.env').resolve()
api_key = open(env_path).readline()

base_command = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
post_command = "epost.fcgi?db=pubmed&retmode=xml&id={}&api_key={}"
summary_command = "esummary.fcgi?db=pubmed&retmode=xml&query_key={}&WebEnv={}&api_key={}"
fetch_command = "efetch.fcgi?db=pubmed&retmode=xml&query_key={}&WebEnv={}&api_key={}"
fetch_full_command = "efetch.fcgi?db=pubmed&retmode=xml&id={}&api_key={}"



def download_data(api_key: str, snippet_list: list):
    list_of_ids = [x["pmid"] for x in snippet_list]
    id_string = ",".join(list_of_ids)

    # post
    post_page = requests.get(base_command + post_command.format(id_string, api_key))
    post_tree = et.ElementTree(et.fromstring(post_page.text))
    post_root = post_tree.getroot()

    web_env = post_root.find('WebEnv').text
    query_key = post_root.find('QueryKey').text
    fetch_page = requests.get(base_command + fetch_command.format(query_key, web_env, api_key))

    fetch_tree = et.ElementTree(et.fromstring(fetch_page.text))
    fetch_root = fetch_tree.getroot()
    pubmed_article_tags = fetch_root.findall('PubmedArticle')

    article_map = {}

    for pubmed_article_tag in pubmed_article_tags:
        article_tag = pubmed_article_tag.find('MedlineCitation').find('Article')
        article_id_tag = pubmed_article_tag.find('PubmedData').find('ArticleIdList').findall('ArticleId')

        pm_id = [x for x in article_id_tag if x.get("IdType") == "pubmed"].pop().text
        title = "" if article_tag is None else article_tag.find('ArticleTitle').text

        try:
            abstract = " ".join(["" if x.text is None else x.text for x in article_tag.find('Abstract').findall('AbstractText')]).strip()
        except AttributeError:
            print('No abstract present - replacing this abstract with " "')
            abstract = ""

        article_map[pm_id] = {"abstract": abstract, "title": title}
    return article_map


def create_json_for_path(path):
    snippet_list = []

    with open(path) as file:
        json_data = json.load(file)

        for q in json_data["questions"]:
            snippets = q["snippets"]

            for snippet in snippets:
                link = snippet["document"]
                bs = snippet["beginSection"]
                pm_id = link[link.rfind("/")+1:]
                snippet_list.append({"pmid": pm_id, "beginSection": bs})

    chunk_length = 200
    s_collection = [snippet_list[x:x + chunk_length] for x in range(0, len(snippet_list), chunk_length)]

    print("There are {} groups of snippets of length {}.".format(len(s_collection), chunk_length))
    return s_collection


if __name__ == "__main__":
    json_file = "training8b.json"
    path_to_data = (base_path / 'BioASQ-training8b').resolve()
    path_to_data_file = (path_to_data / json_file).resolve()

    snippet_list = create_json_for_path(path_to_data_file)
    data_dict = {}

    for snippet_number, snippet in enumerate(tqdm(snippet_list)):
        print("Snippet group: {}".format(snippet_number))
        pm_data = download_data(api_key, snippet)
        data_dict = {**data_dict, **pm_data}

    print(data_dict)

    with open((path_to_data / ("pubmed_" + json_file)).resolve(), 'w') as outfile:
        json.dump(data_dict, outfile)
