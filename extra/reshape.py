with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/data--Hibernate.txt", 'r') as f:
    data_lines = f.readlines()

with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/label--Hibernate.txt", 'r') as g:
    label_lines = g.readlines()

with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/under--Hibernate_1-810.txt", 'r') as p:
    under_lines = p.readlines()

for idx, line in enumerate(under_lines):
    if line.strip() != "None" and line.strip() != "too long.":
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/under--Hivernate_reshaped_.txt", 'a') as t:
            t.write(line)
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/data--Hivernate_reshaped_.txt", 'a') as h:
            h.write(data_lines[idx])
        with open("/Users/yonekuramiki/Desktop/resarch/mydetector/dataset/label--Hivernate_reshaped_.txt", 'a') as j:
            j.write(label_lines[idx])
