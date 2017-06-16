
import json

def main():

    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, multipleChoice_answer, Answer] ... ]
    '''

    train = []
    test = []
    imdir='%s/COCO_%s_%012d.jpg'

    print('Loading annotations and questions...')
    train_anno = json.load(open('annotations/mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open('annotations/mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open('annotations/MultipleChoice_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open('annotations/MultipleChoice_mscoco_val2014_questions.json', 'r'))

    subtype = 'train2014'
    for i in range(len(train_anno['annotations'])):
        ans = train_anno['annotations'][i]['multiple_choice_answer']
        question_id = train_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

        question = train_ques['questions'][i]['question']
        mc_ans = train_ques['questions'][i]['multiple_choices']

        train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans, 'ans': ans})
    
    subtype = 'val2014'
    for i in range(len(val_anno['annotations'])):
        ans = val_anno['annotations'][i]['multiple_choice_answer']
        question_id = val_anno['annotations'][i]['question_id']
        image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

        question = val_ques['questions'][i]['question']
        mc_ans = val_ques['questions'][i]['multiple_choices']

        test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': mc_ans})


    print('Training sample %d, Testing sample %d...' %(len(train), len(test)))

    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))

if __name__ == "__main__":
    main()









