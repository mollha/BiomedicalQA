import pathlib
import xml.etree.ElementTree as et
import gzip


def parse_pm_file_name(name: str):
    prestring = "pubmed20n"
    identifier = name[name.find(prestring) + len(prestring):name.rfind('.xml')]
    return identifier


def find_xml_files(directory: str):
    zipped = list(pathlib.Path(directory).glob('*.xml.gz'))
    xml = list(pathlib.Path(directory).glob('*.xml'))
    return zipped, xml


def parse_xml_file(file_content: str, path_to_csv: str, overwrite=True):
    # Create a new file that can be written to, checking first if it already exists
    if not overwrite and pathlib.Path(path_to_csv).is_file():
        raise FileExistsError("File {} already exists and overwrite flag set to True."
                              "Either change this flag to False or delete the pre-existing file.".format(path_to_csv))

    open(path_to_csv, 'w').close()  # Create an empty file
    csv = open(path_to_csv, "a")  # Open the csv in append mode
    csv.write("text\n")

    tree = et.ElementTree(et.fromstring(file_content))
    root = tree.getroot()
    pubmed_articles = root.findall('PubmedArticle')

    # ---------------- DEFINE STATISTICS -----------------
    total_articles = len(pubmed_articles)
    articles_with_no_title = 0
    articles_with_title_but_no_abstract = 0
    samples_produced = 0

    for idx, article in enumerate(pubmed_articles):
        line_components = []
        article_tag = article.find('MedlineCitation').find('Article')

        # Get the article title
        article_title_tag = article_tag.find('ArticleTitle')

        if article_title_tag is not None:
            title = article_title_tag.text

            if title is not None:
                # Some titles are contained within braces, and terminated by a full-stop.
                if title[0] == "[" and title[-2:] == "].":
                    title = title[1:len(title) - 2]
                line_components.append(title)

            # Get the abstract text title
            abstract_tag = article_tag.find('Abstract')

            if abstract_tag is not None:
                abstract_text_tag = abstract_tag.find('AbstractText')

                if abstract_text_tag is not None and abstract_text_tag.text is not None:
                    line_components.extend([" ", abstract_text_tag.text])
            else:
                articles_with_title_but_no_abstract += 1

            if len(line_components) > 0:
                line_components.append("\n")
                csv.write("".join(line_components))
                samples_produced += 1
        else:
            articles_with_no_title += 1

    csv.close()

    print("\n--------- {} ---------".format(path_to_csv))
    print("Total Articles:", total_articles)
    print("Articles without a Title:", articles_with_no_title)
    print("Articles without an Abstract: ", articles_with_title_but_no_abstract)
    print("Samples Produced: ", samples_produced)
    return articles_with_no_title, articles_with_title_but_no_abstract


if __name__ == "__main__":

    # Process each file in the Dataset directory
    root_data_directory = './Datasets/PubMed'
    raw_data_directory = root_data_directory + "/raw_data"
    processed_data_directory = root_data_directory + "/processed_data"

    zipped_files, xml_files = find_xml_files(raw_data_directory)

    for file in zipped_files:
        f = gzip.open(file, 'rb')
        csv_identifier = processed_data_directory + "/pm_" + parse_pm_file_name(str(file)) + ".csv"
        parse_xml_file(f.read(), path_to_csv=csv_identifier)
        f.close()

    for file in xml_files:
        with open(file, 'r') as f:
            csv_identifier = processed_data_directory + "/pm_" + parse_pm_file_name(str(file)) + ".csv"
            parse_xml_file(f.read(), path_to_csv=csv_identifier)
