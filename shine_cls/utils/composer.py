class SignatureComposer:
    def __init__(self, prompter='a'):
        if prompter not in ['a', 'avg', 'concat', 'isa']:
            raise NameError(f"{prompter} prompter is not supported")

        self._prompter = prompter
        self._composers = {
            'a': self._compose_a,
            'avg': self._compose_avg,
            'concat': self._compose_concat,
            'isa': self._compose_isa,
        }

    def _compose_a(self, signature_list):
        return [f'a {cname}'
                for cname in signature_list]

    def _compose_avg(self, signature_list):
        return [[f'a {catName}' for catName in signature]
                for signature in signature_list]

    def _compose_concat(self, signature_list):
        return ['a ' + signature[0] + ''.join([f' {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def _compose_isa(self, signature_list):
        return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
                for signature in signature_list]

    def compose(self, signature_list):
        return self._composers[self._prompter](signature_list)


if __name__ == '__main__':
    composer = SignatureComposer(prompter='isa')
    signature_list = [
        ['british short hair', 'cat', 'mammal'],
        ['chowchow', 'dog', 'mammal'],
        ['rose', 'flower', 'plant'],
    ]

    result = composer.compose(signature_list)
    print(result)

