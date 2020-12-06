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
    # return zipped, xml
    return zipped, xml


class ParseXMLFiles:
    def __init__(self):
        self.total_articles = 0
        self.articles_with_title_but_no_abstract = 0
        self.abstract_types = {}
        self.abstract_lengths = [[0, 0], [0, 0]]
        self.base_path = pathlib.Path(__file__).parent

    def initiate(self, file_to_parse):
        csv_identifier = processed_data_directory + "/pm_" + parse_pm_file_name(str(file_to_parse)) + ".csv"
        print("Parsing file {}".format(file_to_parse))
        if not overwrite and pathlib.Path(csv_identifier).is_file():
            return
        self.parse_xml_file(f.read(), path_to_csv=(self.base_path / csv_identifier).resolve())

    def parse_xml_file(self, file_content: str, path_to_csv: str):
        # Create a new file that can be written to, checking first if it already exists

        open(path_to_csv, 'w').close()  # Create an empty file
        csv = open(path_to_csv, "a")  # Open the csv in append mode
        csv.write("text\n")

        tree = et.ElementTree(et.fromstring(file_content))
        root = tree.getroot()
        pubmed_articles = root.findall('PubmedArticle')

        # ---------------- DEFINE STATISTICS -----------------
        self.total_articles += len(pubmed_articles)

        for idx, article in enumerate(pubmed_articles):
            line_components = []
            article_tag = article.find('MedlineCitation').find('Article')

            # Get the article title
            article_title_tag = article_tag.find('ArticleTitle')
            title = article_title_tag.text

            if title is not None:
                # Some titles are contained within braces, and terminated by a full-stop.
                if title[0] == "[" and title[-2:] == "].":
                    title = title[1:len(title) - 2]
                line_components.append(title)

            # Get the abstract text title
            abstract_tag = article_tag.find('Abstract')
            other_abstract_tag = article_tag.find('OtherAbstract')
            abstract_elements = [elem for elem in [abstract_tag, other_abstract_tag] if elem is not None]

            if len(abstract_elements) > 0:
                for element in abstract_elements:
                    try:
                        self.abstract_types[element.tag] += 1
                    except KeyError:
                        self.abstract_types[element.tag] = 1

                    abstract_text_tag = element.find('AbstractText')

                    if abstract_text_tag is not None and abstract_text_tag.text is not None:
                        line_components.extend([". ", abstract_text_tag.text])

                        if len(line_components) > 0:
                            joined_line = "".join(line_components)
                            self.abstract_lengths[0][0] += len(joined_line)
                            self.abstract_lengths[0][1] += 1
            else:
                self.articles_with_title_but_no_abstract += 1

                if len(line_components) > 0:
                    joined_line = "".join(line_components)
                    self.abstract_lengths[1][0] += len(joined_line)
                    self.abstract_lengths[1][1] += 1

        csv.close()

    def print_stats(self):
        print("\nTotal Articles:", self.total_articles)
        print("Articles without an Abstract: ", self.articles_with_title_but_no_abstract)
        print("Abstract Types Used: ", self.abstract_types)

        print("\nAverage length of sample without Abstract: ", "%.2f" % (self.abstract_lengths[1][0] / self.abstract_lengths[1][1]))
        print("Average length of sample with Abstract: ", "%.2f" % (self.abstract_lengths[0][0] / self.abstract_lengths[0][1]))


if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent

    # Process each file in the Dataset directory
    root_data_directory = (base_path / './Datasets/PubMed').resolve()
    raw_data_directory = root_data_directory + "/raw_data"
    processed_data_directory = root_data_directory + "/processed_data"
    overwrite = True

    zipped_files, xml_files = find_xml_files(raw_data_directory)
    zipped_files.sort()
    xml_files.sort()

    xml_parser = ParseXMLFiles()

    for file in zipped_files:
        f = gzip.open(file, 'rb')
        xml_parser.initiate(file)
        f.close()

    for file in xml_files:
        with open(file, 'r') as f:
            xml_parser.initiate(file)

    xml_parser.print_stats()

