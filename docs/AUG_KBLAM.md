AutoSchemaKG的三元组一共有三种，以下是一个例子，原文如下：
````
# Operation: Dulce\n\n## Chapter 1\n\nThe thrumming of monitors cast a stark contrast to the rigid silence enveloping the group. Agent Alex Mercer, unfailingly determined on paper, seemed dwarfed by the enormity of the sterile briefing room where Paranormal Military Squad's elite convened. With dulled eyes, he scanned the projectors outlining their impending odyssey into Operation: Dulce., The thrumming of monitors cast a stark contrast to the rigid silence enveloping the group. Agent Alex Mercer, unfailingly determined on paper, seemed dwarfed by the enormity of the sterile briefing room where Paranormal Military Squad's elite convened. With dulled eyes, he scanned the projectors outlining their impending odyssey into Operation: Dulce.\n\n“I assume, Agent Mercer, you’re not having second thoughts?” It was Taylor Cruz’s voice, laced with an edge that demanded attention.\n\nAlex flickered a strained smile, still thumbing his folder's corner. 'Of course not, Agent Cruz. Just trying to soak in all the details.' The compliance in his tone was unsettling, even to himself.\n\nJordan Hayes, perched on the opposite side of the table, narrowed their eyes but offered a supportive nod. 'Details are imperative. We’ll need your clear-headedness down there, Mercer.'\n\nA comfortable silence, the kind that threaded between veterans of shared secrets, lingered briefly before Sam Rivera, never one to submit to quiet, added, 'I’ve combed through the last transmission logs. If anyone can make sense of the anomalies, it’s going to be the two of you.'\n\nTaylor snorted dismissively. 'Focus, people. We have protocols for a reason. Speculation is counter-productive.' The words 'counter-productive' seemed to hang in the air, a tacit reprimand directed at Alex.\n\nFeeling the weight of his compliance conflicting with his natural inclination to leave no stone unturned, Alex straightened in his seat. 'I agree, Agent Cruz. Protocol is paramount,' he said, meeting Taylor's steely gaze. It was an affirmation, but beneath it lay layers of unspoken complexities that would undoubtedly unwind with time.\n\nAlex's submission, though seemingly complete, didn't escape Jordan, who tilted their head ever so slightly, their eyes revealing a spark of understanding. They knew well enough the struggle of aligning personal convictions with overarching missions. As everyone began to collect their binders and prepare for departure, a quiet resolve took form within Alex, galvanized by the groundwork laid by their interactions. He may have spoken in compliance, but his determination had merely taken a subtler form — one that wouldn't surrender so easily to the forthcoming shadows.
````
- 实体-关系-实体

转换成两条KB：
````
key: e1 r1
value: e2

key: e2 r1 by
value: e1
````
````
"entity_relation_dict": [
        {
            "Head": "Alex Mercer",
            "Relation": "was",
            "Tail": "dwarfed"
        },
        {
            "Head": "Taylor Cruz",
            "Relation": "demanded",
            "Tail": "attention"
        },
        {
            "Head": "Alex Mercer",
            "Relation": "offered",
            "Tail": "compliance"
        },
        {
            "Head": "Jordan Hayes",
            "Relation": "offered",
            "Tail": "support"
        },
        {
            "Head": "Sam Rivera",
            "Relation": "added",
            "Tail": "information"
        },
        {
            "Head": "Taylor Cruz",
            "Relation": "reprimanded",
            "Tail": "Alex Mercer"
        },
        {
            "Head": "Alex Mercer",
            "Relation": "aligned",
            "Tail": "protocol"
        },
        {
            "Head": "Jordan Hayes",
            "Relation": "understood",
            "Tail": "Alex Mercer's struggle"
        },
        {
            "Head": "Alex Mercer",
            "Relation": "formed",
            "Tail": "determination"
        }
    ],
````

事件-实体
转换成1+n条KB：
````
key: ev involves 
value: e1, e2, ...

key: e1 involved in
value: ev

key e2 involved in
value: ev
````


````
"event_entity_relation_dict": [
        {
            "Event": "The thrumming of monitors cast a stark contrast to the rigid silence enveloping the group.",
            "Entity": [
                "monitors",
                "group"
            ]
        },
        {
            "Event": "Agent Alex Mercer, unfailingly determined on paper, seemed dwarfed by the enormity of the sterile briefing room where Paranormal Military Squad's elite convened.",
            "Entity": [
                "Agent Alex Mercer",
                "briefing room",
                "Paranormal Military Squad"
            ]
        },
        {
            "Event": "With dulled eyes, he scanned the projectors outlining their impending odyssey into Operation: Dulce.",
            "Entity": [
                "Agent Alex Mercer",
                "projectors",
                "Operation: Dulce"
            ]
        },
        {
            "Event": "It was Taylor Cruz’s voice, laced with an edge that demanded attention.",
            "Entity": [
                "Taylor Cruz"
            ]
        },
        {
            "Event": "Alex flickered a strained smile, still thumbing his folder's corner.",
            "Entity": [
                "Agent Alex Mercer"
            ]
        },
        {
            "Event": "'Of course not, Agent Cruz. Just trying to soak in all the details.'",
            "Entity": [
                "Agent Alex Mercer",
                "Agent Cruz"
            ]
        },
        {
            "Event": "Jordan Hayes, perched on the opposite side of the table, narrowed their eyes but offered a supportive nod.",
            "Entity": [
                "Jordan Hayes"
            ]
        },
        {
            "Event": "'Details are imperative. We’ll need your clear-headedness down there, Mercer.'",
            "Entity": [
                "Jordan Hayes",
                "Agent Alex Mercer"
            ]
        },
        {
            "Event": "A comfortable silence, the kind that threaded between veterans of shared secrets, lingered briefly.",
            "Entity": [
                "group"
            ]
        },
        {
            "Event": "Sam Rivera, never one to submit to quiet, added, 'I’ve combed through the last transmission logs. If anyone can make sense of the anomalies, it’s going to be the two of you.'",
            "Entity": [
                "Sam Rivera",
                "Agent Alex Mercer",
                "Taylor Cruz"
            ]
        },
        {
            "Event": "Taylor snorted dismissively.",
            "Entity": [
                "Taylor Cruz"
            ]
        },
        {
            "Event": "'Focus, people. We have protocols for a reason. Speculation is counter-productive.'",
            "Entity": [
                "Taylor Cruz"
            ]
        },
        {
            "Event": "The words 'counter-productive' seemed to hang in the air, a tacit reprimand directed at Alex.",
            "Entity": [
                "Taylor Cruz",
                "Agent Alex Mercer"
            ]
        },
        {
            "Event": "Feeling the weight of his compliance conflicting with his natural inclination to leave no stone unturned, Alex straightened in his seat.",
            "Entity": [
                "Agent Alex Mercer"
            ]
        },
        {
            "Event": "'I agree, Agent Cruz. Protocol is paramount,' he said, meeting Taylor's steely gaze.",
            "Entity": [
                "Agent Alex Mercer",
                "Agent Cruz",
                "Taylor Cruz"
            ]
        },
        {
            "Event": "It was an affirmation, but beneath it lay layers of unspoken complexities that would undoubtedly unwind with time.",
            "Entity": [
                "Agent Alex Mercer"
            ]
        },
        {
            "Event": "Alex's submission, though seemingly complete, didn't escape Jordan, who tilted their head ever so slightly, their eyes revealing a spark of understanding.",
            "Entity": [
                "Agent Alex Mercer",
                "Jordan Hayes"
            ]
        },
        {
            "Event": "They knew well enough the struggle of aligning personal convictions with overarching missions.",
            "Entity": [
                "Jordan Hayes"
            ]
        },
        {
            "Event": "As everyone began to collect their binders and prepare for departure, a quiet resolve took form within Alex, galvanized by the groundwork laid by their interactions.",
            "Entity": [
                "Agent Alex Mercer",
                "group"
            ]
        },
        {
            "Event": "He may have spoken in compliance, but his determination had merely taken a subtler form — one that wouldn't surrender so easily to the forthcoming shadows.",
            "Entity": [
                "Agent Alex Mercer"
            ]
        }
    ],
````

事件-关系-事件
转换成1条KB：
````
key: ev1 happens r1
value: ev2
````
(后续如果考虑添加逆转关系的话就需要生成每个关系的逆转关系)

````
"event_relation_dict": [
        {
            "Head": "The thrumming of monitors cast a stark contrast to the rigid silence enveloping the group.",
            "Relation": "before",
            "Tail": "Agent Alex Mercer, unfailingly determined on paper, seemed dwarfed by the enormity of the sterile briefing room where Paranormal Military Squad's elite convened."
        },
        {
            "Head": "Agent Alex Mercer, unfailingly determined on paper, seemed dwarfed by the enormity of the sterile briefing room where Paranormal Military Squad's elite convened.",
            "Relation": "at the same time",
            "Tail": "With dulled eyes, he scanned the projectors outlining their impending odyssey into Operation: Dulce."
        },
        {
            "Head": "It was Taylor Cruz’s voice, laced with an edge that demanded attention.",
            "Relation": "because",
            "Tail": "Alex flickered a strained smile, still thumbing his folder's corner."
        },
        {
            "Head": "Alex flickered a strained smile, still thumbing his folder's corner.",
            "Relation": "because",
            "Tail": "The compliance in his tone was unsettling, even to himself."
        },
        {
            "Head": "Jordan Hayes, perched on the opposite side of the table, narrowed their eyes but offered a supportive nod.",
            "Relation": "at the same time",
            "Tail": "'Details are imperative. We’ll need your clear-headedness down there, Mercer.'"
        },
        {
            "Head": "'Details are imperative. We’ll need your clear-headedness down there, Mercer.'",
            "Relation": "because",
            "Tail": "A comfortable silence, the kind that threaded between veterans of shared secrets, lingered briefly before Sam Rivera, never one to submit to quiet, added, 'I’ve combed through the last transmission logs. If anyone can make sense of the anomalies, it’s going to be the two of you.'"
        },
        {
            "Head": "A comfortable silence, the kind that threaded between veterans of shared secrets, lingered briefly before Sam Rivera, never one to submit to quiet, added, 'I’ve combed through the last transmission logs. If anyone can make sense of the anomalies, it’s going to be the two of you.'",
            "Relation": "because",
            "Tail": "Taylor snorted dismissively. 'Focus, people. We have protocols for a reason. Speculation is counter-productive.'"
        },
        {
            "Head": "Taylor snorted dismissively. 'Focus, people. We have protocols for a reason. Speculation is counter-productive.'",
            "Relation": "because",
            "Tail": "The words 'counter-productive' seemed to hang in the air, a tacit reprimand directed at Alex."
        },
        {
            "Head": "Feeling the weight of his compliance conflicting with his natural inclination to leave no stone unturned, Alex straightened in his seat.",
            "Relation": "because",
            "Tail": "It was an affirmation, but beneath it lay layers of unspoken complexities that would undoubtedly unwind with time."
        },
        {
            "Head": "Alex's submission, though seemingly complete, didn't escape Jordan, who tilted their head ever so slightly, their eyes revealing a spark of understanding.",
            "Relation": "because",
            "Tail": "They knew well enough the struggle of aligning personal convictions with overarching missions."
        },
        {
            "Head": "As everyone began to collect their binders and prepare for departure, a quiet resolve took form within Alex, galvanized by the groundwork laid by their interactions.",
            "Relation": "as a result",
            "Tail": "He may have spoken in compliance, but his determination had merely taken a subtler form — one that wouldn't surrender so easily to the forthcoming shadows."
        }
    ],

````


# 公共数据集-RAGBench

子数据集(训练条数)：
    finqa 12502条
    hagrid 2892条
    tatqa 26430
提取数据集中的相关列：question, document, response


## 三元组提取

将数据集转换成AutoSchemaKG接受的格式
    id:
    text:
    metadata:
        language: en

````bash
python 2.datasets_prepare.py multi_wiki_qa /mnt/n0/datasets/multi-wiki-qa.json ../example_data/multi_wiki_qa.json
````



运行AutoSchemaKG提取三元组
````bash
export CUDA_VISIBLE_DEVICES=0,1
python 3.triple_gen.py


将提取的三元组转换统一格式用于训练
````
python 4.traiple_reformat.py ../import/multi_wiki_qa_train/kg_extraction/_mnt_n0_models_llama3_8B_instruct__multi_wiki_qa_train_output_20251009162046_1_in_1.json ../import/multi_wiki_qa_train/kg_extraction/multi_wiki_qa_train.json
````
