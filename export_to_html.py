import os

for file in os.listdir('.'):
    if file.endswith('.md'):
        filename = os.path.splitext(os.path.basename(file))[0]
        command_line = 'pandoc -f gfm -t html5 --mathjax --katex --mathml -s "'  + file + '" -o "html/' + filename + '.html"'
        print(command_line)
        os.system(command_line)
