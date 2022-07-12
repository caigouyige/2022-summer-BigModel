import bminf
from tqdm import tqdm


def story_generator(model, description):
    if description == '':
        sentence = "讲个故事吧："
        with tqdm() as progress_bar:
            stoped = False
            while not stoped:
                sub_sentence, stoped = model.generate(
                    sentence, 
                    max_tokens=8,
                    top_n=5,
                    top_p=None,
                    temperature=0.85,
                    frequency_penalty=0,
                    presence_penalty=1
                )
                sentence += sub_sentence
                progress_bar.update(1)
        print(sentence)
    else:
        
        sentence = "一个" + description+ "故事："
        with tqdm() as progress_bar:
            stoped = False
            while not stoped:
                sub_sentence, stoped = model.generate(
                    sentence, 
                    max_tokens=8,
                    top_n=5, 
                    top_p=None,
                    temperature=0.85,
                    frequency_penalty=0,
                    presence_penalty=1
                )
                sentence += sub_sentence
                progress_bar.update(1)
        print(sentence)
    

def main():
    description = input("请输入一个词语，描述你想生成故事类型（若直接回车，则会随机生成一个故事）：")
    print("正在生成故事...")
    cpm1 = bminf.models.CPM1()
    story_generator(cpm1, description)

if __name__ == "__main__":
    main()