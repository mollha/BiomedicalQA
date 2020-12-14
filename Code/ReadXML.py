import pathlib
import xml.etree.ElementTree as et
import gzip
import random


def parse_pm_file_name(name: str):
    prestring = "pubmed20n"
    identifier = name[name.find(prestring) + len(prestring):name.rfind('.xml')]
    return identifier


def find_xml_files(directory):
    zipped = list(directory.glob('*.xml.gz'))
    xml = list(directory.glob('*.xml'))

    # Shuffle the lists of files
    random.shuffle(zipped)
    random.shuffle(xml)
    return zipped, xml


class ParseXMLFiles:
    def __init__(self, max_dataset_size, max_samples_per_file):
        self.max_dataset_size = max_dataset_size
        self.max_samples_per_file = max_samples_per_file

        print("Creating dataset with max dataset size of {} and max samples per file of {}"
              .format(max_dataset_size, max_samples_per_file))

        self.total_articles = 0
        self.articles_parsed = 0

        self.abstract_types = {}
        self.abstract_lengths = [0, 0]
        self.base_path = pathlib.Path(__file__).parent

    def initiate(self, file_name, file):
        csv_identifier = str(processed_data_directory) + "/pm_" + parse_pm_file_name(str(file_name)) + ".csv"
        print("Parsing file {}".format(file_name))
        if not overwrite and pathlib.Path(csv_identifier).is_file():
            return

        self.parse_xml_file(file.read(), path_to_csv=(self.base_path / csv_identifier).resolve())

    def parse_xml_file(self, file_content: str, path_to_csv: str):
        # Create a new file that can be written to, checking first if it already exists

        open(path_to_csv, 'w').close()  # Create an empty file
        csv = open(path_to_csv, "a")  # Open the csv in append mode
        csv.write("text\n")

        tree = et.ElementTree(et.fromstring(file_content))
        root = tree.getroot()
        pubmed_articles = root.findall('PubmedArticle')
        random.shuffle(pubmed_articles)

        # ---------------- DEFINE STATISTICS -----------------
        self.total_articles += len(pubmed_articles)
        samples_so_far = 0
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
                        line_components.extend([" ", abstract_text_tag.text])

                        if len(line_components) > 0:
                            if samples_so_far < self.max_samples_per_file and self.articles_parsed < self.max_dataset_size:
                                samples_so_far += 1
                                self.articles_parsed += 1
                                joined_line = "".join(line_components)
                                self.abstract_lengths[0] += len(joined_line)
                                self.abstract_lengths[1] += 1
                                csv.write(joined_line + "\n")
                            break

        print("{} samples produced from this file. {} articles parsed in total - max dataset size is {}."
              .format(samples_so_far, self.articles_parsed, self.max_dataset_size))
        csv.close()

    def print_stats(self):
        print("\nTotal Articles:", self.total_articles)
        print("Abstract Types Used: ", self.abstract_types)
        print("Average length of sample with Abstract: ", "%.2f" % (self.abstract_lengths[0] / self.abstract_lengths[1]))


# find way to collect dataset stats
if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent
    max_dataset_size = 14000000

    # Process each file in the Dataset directory
    raw_data_directory = (base_path / './datasets/PubMed/raw_data').resolve()
    processed_data_directory = (base_path / './datasets/PubMed/processed_data').resolve()
    overwrite = True

    zipped_files, xml_files = find_xml_files(raw_data_directory)

    max_samples_per_file = int((max_dataset_size // (len(zipped_files) + len(xml_files))) + 1)
    xml_parser = ParseXMLFiles(max_dataset_size, max_samples_per_file)

    for file in zipped_files:
        f = gzip.open(file, 'rb')
        xml_parser.initiate(str(file), f)
        f.close()

    for file in xml_files:
        with open(file, 'r') as f:
            xml_parser.initiate(str(file), f)

    xml_parser.print_stats()

