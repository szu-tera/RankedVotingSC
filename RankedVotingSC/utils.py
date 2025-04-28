import json

class Converter:
    def __init__(self):
        pass

    # todo
    def json2jsonl(self, input_file, output_file):
        assert input_file.endswith('.json'), f"ERROR: {input_file} is not a JSON file!"
        if output_file is not None:
            assert output_file.endswith('.jsonl'), f"ERROR: {output_file} is not a JSONL file!"
        else:
            output_file = input_file.replace('.json', '.jsonl')

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            for item in data:
                json_line = json.dumps(item, ensure_ascii=False)
                f_out.write(json_line + '\n')
        print(f"Convert {input_file} to {output_file}")

    # todo
    def jsonl2json(self, input_file, output_file=None):
        assert input_file.endswith('.jsonl'), f"ERROR: {input_file} is not a JSONL file!"
        if output_file is not None: 
            assert output_file.endswith('.json'), f"ERROR: {input_file} is not a JSON file!"
        else:
            output_file = input_file.replace('.jsonl', '.json')
        
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)
        print(f"Convert {input_file} to {output_file}")