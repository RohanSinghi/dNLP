from torchtext.data import Field


class TextField(Field):
    """Defines a common text field.

    Attributes:
        is_target (bool): Whether this field represents a target variable.
    """

    def __init__(self, **kwargs):
        """Initializes a TextField object.

        Args:
            sequential (bool): Whether the field is sequential. Default: True.
            use_vocab (bool): Whether to use a vocabulary. Default: True.
            init_token (str): A token that will be prepended to any example
                using this field, or None for no initial token. Default: None.
            eos_token (str): A token that will be appended to any example
                using this field, or None for no end-of-sentence token. Default: None.
            fix_length (int): A fixed length that all examples using this field will
                be padded to, or None for flexible sequence lengths. Default: None.
            dtype (torch.dtype): The data type of the Tensors created by this field.
                Default: torch.long.
            preprocessing (Pipeline or List[Pipeline]): The Pipeline (or list of
                Pipelines) that will be applied to examples using this field after
                tokenizing but before numericalizing. Default: None.
            postprocessing (Pipeline or List[Pipeline]): A Pipeline (or list of
                Pipelines) that will be applied to examples using this field after
                numericalizing but before the numbers are turned into a Tensor.
                Default: None.
            lower (bool): Whether to lowercase the text. Default: False.
            tokenize (callable): The function used to tokenize strings. By default, we use
                ``lambda x: x.split()`` when possible. If a more complex
                tokenization is required, the tokenize argument should be used.
                Default: str.split.
            include_lengths (bool): Whether to return a tuple of a padded sequence and
                a list of lengths. Default: False.
            batch_first (bool): Whether to produce tensors with the batch dimension
                as the first dimension. Default: False.
            pad_token (str): The string used as padding token. Default: "<pad>".
            pad_first (bool): Do the padding of the sequence at the beginning. Default: False.
            truncate_first (bool): Do the truncating of the sequence at the beginning. Default: False.
            is_target (bool): Whether this field represents a target variable. Default: False.
        """
        self.is_target = kwargs.pop('is_target', False)
        super().__init__(**kwargs)


class LabelField(Field):
    """Defines a label field.

    Attributes:
        is_target (bool): Whether this field represents a target variable.
    """

    def __init__(self, **kwargs):
        """Initializes a LabelField object.

        Args:
            sequential (bool): Whether the field is sequential. Default: False.
            use_vocab (bool): Whether to use a vocabulary. Default: True.
            init_token (str): A token that will be prepended to any example
                using this field, or None for no initial token. Default: None.
            eos_token (str): A token that will be appended to any example
                using this field, or None for no end-of-sentence token. Default: None.
            fix_length (int): A fixed length that all examples using this field will
                be padded to, or None for flexible sequence lengths. Default: None.
            dtype (torch.dtype): The data type of the Tensors created by this field.
                Default: torch.long.
            preprocessing (Pipeline or List[Pipeline]): The Pipeline (or list of
                Pipelines) that will be applied to examples using this field after
                tokenizing but before numericalizing. Default: None.
            postprocessing (Pipeline or List[Pipeline]): A Pipeline (or list of
                Pipelines) that will be applied to examples using this field after
                numericalizing but before the numbers are turned into a Tensor.
                Default: None.
            lower (bool): Whether to lowercase the text. Default: False.
            tokenize (callable): The function used to tokenize strings. By default, we use
                ``lambda x: x.split()`` when possible. If a more complex
                tokenization is required, the tokenize argument should be used.
                Default: str.split.
            include_lengths (bool): Whether to return a tuple of a padded sequence and
                a list of lengths. Default: False.
            batch_first (bool): Whether to produce tensors with the batch dimension
                as the first dimension. Default: False.
            pad_token (str): The string used as padding token. Default: "<pad>".
            pad_first (bool): Do the padding of the sequence at the beginning. Default: False.
            truncate_first (bool): Do the truncating of the sequence at the beginning. Default: False.
            is_target (bool): Whether this field represents a target variable. Default: False.
        """
        kwargs['sequential'] = False
        self.is_target = kwargs.pop('is_target', True)
        super().__init__(**kwargs)