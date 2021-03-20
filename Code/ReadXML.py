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
    def __init__(self, processed_data_directory, max_dataset_size, max_samples_per_file):
        self.processed_data_directory = processed_data_directory
        self.current_csv = None

        self.csv_suffix = 1
        self.max_dataset_size = max_dataset_size
        self.max_samples_per_file = max_samples_per_file

        print("Creating dataset with max dataset size of {} and max samples per file of {}.\n"
              .format(max_dataset_size, max_samples_per_file))

        self.samples_in_file = 0
        self.total_articles = 0
        self.articles_parsed = 0
        self.abstract_lengths = [0, 0]
        self.base_path = pathlib.Path(__file__).parent

    def create_new_csv(self):
        """
        Given a CSV suffix, create a file with a valid name and write the header line.
        :return: None
        """
        string_suffix = str(self.csv_suffix).zfill(4)
        csv_identifier = self.processed_data_directory + "/pm_" + string_suffix + ".csv"
        print("\nCreating file '{}'".format("pm_" + string_suffix + ".csv"))

        path_to_csv = (self.base_path / csv_identifier).resolve()

        open(path_to_csv, 'w').close()  # Create an empty file
        csv = open(path_to_csv, "a")  # Open the csv in append mode
        csv.write("text\n")
        csv.close()

        self.csv_suffix += 1
        self.samples_in_file = 0
        self.current_csv = path_to_csv

    def write_line(self, line_components, csv):
        self.samples_in_file += 1
        self.articles_parsed += 1

        for idx in range(len(line_components)):
            component = line_components[idx]
            if "\n" in component:
                line_components[idx] = component.replace("\n", " ")

        joined_line = "".join(line_components)

        self.abstract_lengths[0] += len(joined_line)
        self.abstract_lengths[1] += 1
        csv.write(joined_line + "\n")

    def parse_xml_file(self, file_content: str):
        if self.articles_parsed >= self.max_dataset_size:
            return False

        if self.current_csv is None:
            self.create_new_csv()

        # Create a new file that can be written to, checking first if it already exists
        csv = open(self.current_csv, "a")  # Open the csv in append mode

        tree = et.ElementTree(et.fromstring(file_content))
        root = tree.getroot()
        pubmed_articles = root.findall('PubmedArticle')
        random.shuffle(pubmed_articles)

        # ---------------- DEFINE STATISTICS -----------------
        self.total_articles += len(pubmed_articles)

        for idx, article in enumerate(pubmed_articles):
            if self.articles_parsed >= self.max_dataset_size:
                return False

            line_components = []

            article_tag = article.find('MedlineCitation').find('Article')
            article_title_tag = article_tag.find('ArticleTitle')
            title = article_title_tag.text

            if title is not None:
                # Some titles are contained within braces, and terminated by a full-stop.
                if title[0] == "[" and title[-2:] == "].":
                    title = title[1:len(title) - 2]
                    title = title.strip()
                    # add full-stop if there isn't one
                    if title[-1] != ".":
                        title += "."
                line_components.append(title)

            # Get the abstract text title
            abstract_tag = article_tag.find('Abstract')
            other_abstract_tag = article_tag.find('OtherAbstract')
            abstract_elements = [elem for elem in [abstract_tag, other_abstract_tag] if elem is not None]

            if len(abstract_elements) > 0:
                for element in abstract_elements:
                    abstract_text_tag = element.find('AbstractText')

                    if abstract_text_tag is not None and abstract_text_tag.text is not None:
                        line_components.extend([" ", abstract_text_tag.text])

                        if len(line_components) > 0:
                            if self.samples_in_file == self.max_samples_per_file:
                                print("File '{}' contains {} samples - {} samples constructed in total."
                                      .format(str(self.current_csv)[-8:], self.samples_in_file,
                                              self.articles_parsed))
                                self.create_new_csv()
                                csv.close()
                                csv = open(self.current_csv, "a")

                            if self.samples_in_file < self.max_samples_per_file:
                                self.write_line(line_components, csv)
                            break
        csv.close()
        return True

    def print_stats(self):
        print("\nTotal Articles:", self.total_articles)
        print("Average length of sample with Abstract: ", "%.2f" % (self.abstract_lengths[0] / self.abstract_lengths[1]))


# find way to collect dataset stats
if __name__ == "__main__":
    base_path = pathlib.Path(__file__).parent
    max_dataset_size = 30000000

    # Process each file in the Dataset directory
    raw_data_directory = (base_path / './datasets/PubMed/raw_data').resolve()
    processed_data_directory = (base_path / './datasets/PubMed/processed_data').resolve()
    zipped_files, xml_files = find_xml_files(raw_data_directory)

    # max_samples_per_file = int((max_dataset_size // (len(zipped_files) + len(xml_files))) + 1)
    max_samples_per_file = 8000

    xml_parser = ParseXMLFiles(str(processed_data_directory), max_dataset_size, max_samples_per_file)

    for file in zipped_files:
        f = gzip.open(file, 'rb')

        print("Opening file {}".format(str(file)))
        response = xml_parser.parse_xml_file(f.read())
        f.close()
        if not response:
            break

    for file in xml_files:
        with open(file, 'r') as f:
            print("Parsing file {}".format(str(file)))

            try:
                response = xml_parser.parse_xml_file(f.read())
            except Exception:
                # ignore this file
                continue

            if not response:
                break

    xml_parser.print_stats()
