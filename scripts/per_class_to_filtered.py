import json
import argparse

from data.utils import format_concept


def main(per_class_file_path, filtered_file_path, filtered_classses):
    with open(per_class_file_path) as f:
        per_class_data = json.load(f)

    all_concepts = []
    classes = list(per_class_data.keys())
    for key in per_class_data.keys():
        all_concepts.extend(per_class_data[key])

    all_concepts = [format_concept(concept) for concept in all_concepts]

    with open(filtered_file_path, 'w') as f:
        for concept in all_concepts:
            f.write(concept + '\n')

    with open(filtered_classses, 'w') as f:
        for concept in classes:
            f.write(concept + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_class_file_path', type=str, required=True)
    parser.add_argument('--filtered_file_path', type=str, required=True)
    parser.add_argument('--filtered_classses', type=str, required=True)
    args = parser.parse_args()
    main(args.per_class_file_path, args.filtered_file_path, args.filtered_classses)
