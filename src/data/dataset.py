from torchtext.legacy import data
from src.data.fields import TextField, LabelField


class NLPDataset(data.Dataset):
    """Defines a base class for NLP datasets.

    Attributes:
        fields (dict[str, Field]): Dictionary mapping field names to Field objects.
        examples (list[Example]): List of Example objects.
    """

    def __init__(self, examples, fields, **kwargs):
        """Initializes a NLPDataset object.

        Args:
            examples (list[Example]): List of Example objects.
            fields (dict[str, Field]): Dictionary mapping field names to Field objects.
        """
        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, fields, root='.data', train='train.txt', validation='val.txt', test='test.txt', **kwargs):
        """Create train, validation, and test datasets given path."

        train_data = None if train is None else cls(root=root, train=train, fields=fields, **kwargs)
        val_data = None if validation is None else cls(root=root, validation=validation, fields=fields, **kwargs)
        test_data = None if test is None else cls(root=root, test=test, fields=fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

    @classmethod
    def from_file(cls, path, format, fields, **kwargs):
        """Create dataset from a file.

        Args:
            path (str): Path to the data file.
            format (str|dict): Format of the data file.  If str, it should be one of 'csv', 'tsv', or 'json'.  If dict, it should be
                a dictionary mapping field names to column indices.
            fields (dict[str, Field]): Dictionary mapping field names to Field objects.
        """
        with open(path, encoding='utf-8') as f:
            if isinstance(format, str):
                if format == 'csv':
                    import csv
                    reader = csv.reader(f)
                    header = next(reader)
                    examples = []
                    for row in reader:
                        example = data.Example.fromlist([row[i] for i in range(len(row))], fields)
                        examples.append(example)
                elif format == 'tsv':
                    import csv
                    reader = csv.reader(f, delimiter='\t')
                    header = next(reader)
                    examples = []
                    for row in reader:
                        example = data.Example.fromlist([row[i] for i in range(len(row))], fields)
                        examples.append(example)
                elif format == 'json':
                    import json
                    data = json.load(f)
                    examples = []
                    for item in data:
                        example = data.Example.fromdict(item, fields)
                        examples.append(example)
                else:
                    raise ValueError("Invalid format: {}".format(format))
            elif isinstance(format, dict):
                examples = []
                for line in f:
                    items = line.strip().split()
                    example = data.Example.fromlist([items[i] for i in format.values()], fields)
                    examples.append(example)
            else:
                raise TypeError("format must be str or dict, not {}".format(type(format)))

        return cls(examples, fields, **kwargs)


if __name__ == '__main__':
    # Example usage of the custom Field classes
    TEXT = TextField(tokenize='spacy', lower=True)
    LABEL = LabelField(dtype=data.torch.float)

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

    # Create dummy data
    examples = [
        data.Example.fromlist(['This is a positive example.', '1'], fields),
        data.Example.fromlist(['This is a negative example.', '0'], fields),
        data.Example.fromlist(['Another positive one.', '1'], fields)
    ]

    # Create a dataset
    dataset = NLPDataset(examples, fields)

    # Print some information
    print(f'Number of examples: {len(dataset)}')
    print(f'Fields: {dataset.fields}')
    print(f'Example 0 text: {dataset[0].text}')
    print(f'Example 0 label: {dataset[0].label}')