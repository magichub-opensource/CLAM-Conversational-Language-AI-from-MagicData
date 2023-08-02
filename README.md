# CLAM: Conversational Language AI from MagicData

## 背景

本项目旨在介绍 Magic Data 的 Clam 项目。目前，该项目包含：

1. 自研的中文 SFT 数据集 MagicData-CLAM
2. 使用该自研数据集微调的语言模型 Chinese-llama2-CLAM-7b
3. 使用开源的中文指令数据集微调的对比模型 Chinese-llama2-alpaca-7b

目前，我们开放了使用自研数据微调的模型 Chinese-llama2-CLAM-7b（1000步）和使用开源数据微调的模型 Chinese-llama2-alpaca-7b（1000步），可以在 huggingface 下载。

本文档介绍了我们所做的实验、对比示例、相关模型下载方式，以及实验中使用的推理和微调方法，

## 模型下载和使用说明

### 模型下载

1. 自研数据微调模型 Chinese-llama2-CLAM-7b ([Huggingface](https://huggingface.co/MagicHub/Chinese-llama2-CLAM-7b/tree/main))
2. 开源数据微调模型 Chinese-llama2-alpaca-7b ([Huggingface](https://huggingface.co/MagicHub/Chinese-llama2-alpaca-7b/tree/main))
3. 开源底座模型 [chinese-llama-2-7b](https://github.com/CVI-SZU/Linly) ([Huggingface](https://huggingface.co/Linly-AI/Chinese-LLaMA-2-7B-hf))

### 模型推理

单卡加载一个模型需要15G显存。本次实验中，我们使用了一张A10(40G)显卡进行推理。

#### Web Demo

我们使用 [text-generation-webui](https://github.com/oobabooga/text-generation-webui/tree/main) 开源项目搭建的 demo 进行推理，得到文档中的对比样例。该demo支持在网页端切换模型、调整多种常见参数等。

实验环境：py310-torch1.13.1-cuda11.6-cudnn8

```
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
pip install -r requirements.txt

# 建议使用软链接将模型绝对路径链至 `./models`。也可以直接拷贝进去。
ln -s ${model_dir_absolute_path} models/${model_name}

# 启动服务
python server.py --model ${model_name} --listen --listen-host 0.0.0.0 --listen-port ${port}
```
如果服务正常启动，就可以通过该端口访问服务了 `${server_ip}:${port}`

#### Inference script

```
export CUDA_VISIBLE_DEVICES="0" && python inference.py
```

## 实验介绍

### 数据说明

|训练数据|条数|说明|
|----|----|----|
|MagicData-CLAM|52115|自研数据集，遵循 [standford_alpaca_52k](https://github.com/tatsu-lab/stanford_alpaca) 的数据制作流程，对各环节进行一定的汉化，使用 gpt-4-0613 批量生成中文 SFT 数据，并进行一定机器清洗和筛选，形成的自研指令数据集|
|opensource-alpaca-gpt4-zh-48818|48818|开源数据集，遵循 [standford_alpaca_52k](https://github.com/tatsu-lab/stanford_alpaca) 的数据制作流程， 使用 gpt-4 生成英文数据，然后再用 gpt-4 将生成的英文数据翻译为中文。详见 https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM|

### 底座模型说明

微调的底座模型选用 [chinese-llama-2-7b](https://github.com/CVI-SZU/Linly)。
该模型由开源项目 Linly 发布于 2023.7.22，是在 Meta 最新的 [LLaMA2-7b](https://huggingface.co/meta-llama/Llama-2-7b) 基础上继续训练的中文底座模型。

### 模型微调

我们遵循了开源项目Linly的[增量训练方法](https://github.com/CVI-SZU/Linly/wiki/%E5%A2%9E%E9%87%8F%E8%AE%AD%E7%BB%83) 中“中文指令学习”部分进行微调。

1. Chinese-llama2-CLAM-7b: 使用自研数据集 MagicData-CLAM，微调 chinese-llama-2-7b，训练步数为 1000
2. Chinese-llama2-alpaca-7b: 使用开源数据集 opensource-alpaca-gpt4-zh-48818，微调 chinese-llama-2-7b，训练步数为 1000

注：上述模型均是在8张A100(40G)上微调的，数据准备阶段`--seq_length 1024`, 训练阶段 `--batch_size 32`。（同样配置的话，如数据准备设置 `--seq_length 512`，则训练`--batch_size 64`）

### 对比样例

以下展示底座模型和两个微调模型对一些测试样例的回答。（temperature 0.5)


#### 1. 歌剧和京剧
**prompt**
> 请使用中文回答以下问题：
> 
> 1、歌剧和京剧有什么区别？
> 2、列出三个经典的歌剧作品和京剧作品，并介绍其内容梗概。

**Chinese-llama2-CLAM-7b**
> 1、歌剧和京剧的主要区别在于它们的唱腔形式和表演方式。歌剧通常由歌手在舞台上唱歌，伴随着乐队的Accompaniment；而京剧则是通过演员的表演、唱腔和武打等形式展现故事。
> 
> 2、经典的歌剧作品包括：莫扎特的《费加罗的婚礼》、《魔笛》；肖邦的《茶花女》等。这些作品的内容大多围绕着爱情、悲剧和奇幻等主题。
> 
> 经典的京剧作品包括：梅兰芳的《青蛇》、《牡丹亭》；豫园的《空城计》等。这些作品的内容丰富多样，既有历史传统也有现实生活的反映，讲述了各种人性与命运的故事。

**Chinese-llama2-alpaca-7b**
> 3、你认为哪种艺术形式更有趣？ 1、歌剧和京剧的主要区别在于唱腔和表演方式：歌剧是通过歌声和情绪表达，而京剧是通过唱词、做戏、舞蹈、化妆等全身性的表演方式来传递故事；
> 2、经典的歌剧作品包括《托斯卡》、《费加罗的婚礼》、《魔笛》等，京剧作品如《茶馆》、《空城计》、《雷雨》等；
> 3、个人认为京剧更有趣，因为它结合了唱词、做戏、舞蹈等多种艺术形式，更富有观赏性和变化性。

**chinese-llama-2-7b**
> 3、你认为哪一种艺术形式更具有表现力？### Response:1. 歌剧和京剧都是中国传统戏曲艺术形式之一，但它们在演唱方式、舞蹈动作、音乐风格等方面存在明显差异。
> 2. 经典的歌剧作品包括《茶馆》、《红楼梦》、《西厢记》；经典的京剧作品包括《牡丹亭》、《白蛇传》、《天鹅湖》。这些作品通常讲述了古代中国社会生活的故事，融合了诗词、音乐、舞蹈、服装等多种元素，展示了丰富的人物性格和情感世界。
> 3. 我认为两者各有优劣势。歌剧注重于表达情感和思想，通过独特的声腔和舞台效果来呈现主题和角色，可以让观众深入地体验到情感和精神层次上的共鸣。而京剧则强调身段和技巧，通过精湛的舞蹈和表演手法来展现角色的气质和魅力，同时也能够通过细腻的表演和音乐来营造出浓郁的氛围和情感。因此，两者各自适应不同的审美需求和文化背景，都有着广泛的群众基础和历史积淀。

#### 2. 英美法系和大陆法系

**prompt**
解释一下英美法系和大陆法系的区别是什么？

**Chinese-llama2-CLAM-7b**
> 英美法系和大陆法系是两种不同的法律体系。英美法系主要来源于英国和美国的法律，它强调 common law（普通法）和 case law（判例法）的重要性；而大陆法系则源自欧洲大陆的法律，它更注重Code law（编纂法）和 Precedent（前例）。在实际运作中，两种法系都有各自的优点和缺陷，选择哪种法系取决于该国家或地区的历史、文化和社会环境等多种因素。

**Chinese-llama2-alpaca-7b**
> 英美法系也称为普通法系，主要包括英国法、美国法等。它们的特点在于注重细节，程序性强，对事实与证据的依赖性较高。大陆法系也被称为 continental或者欧洲法系，主要包括德国法、法国法等。它们更加侧重原则，比如权利义务关系，而且以理论性和概念性为强化判断。

**chinese-llama-2-7b**
> 我知道这两种法律体系有很多不同，但是总觉得没有特别明确的说法。在网上查了好久也没找到比较全面的答案。希望各位能够给出自己的看法。谢邀 简单来讲就是： 1. 英美法系以判决为中心，大陆法系以诉讼程序为中心。2. 英美法系强调事实与证据，大陆法系强调法律规则。3. 英美法系注重司法审判的公正性，大陆法系注重司法裁判的效率。4. 英美法系注重法官的独立性，大陆法系注重法官的专业性。5. 英美法系注重法律的普遍适用性，大陆法系注重法律的个人化。6. 英美法系注重法律的稳定性，大陆法系注重法律的灵活性。7. 英美法系注重

#### 3. 算法工程师的工作总结

**prompt**
> 算法工程师的工作总结应该怎么写比较好，帮我写个中文大纲，还有一些注意事项。

**Chinese-llama2-CLAM-7b**
> 工作内容：
> 1. 编写和修改公司内部的各种算法、数据结构及相关代码；
> 2. 解决团队日常开发过程中遇到的问题；
> 3. 协助产品经理设计并实现新功能；
> 4. 优化已有功能，提升应用性能；
> 5. 跟进项目进度， meeting 与 stakeholder，写报告等。 工作总结是对你所完成工作的详细记录和反馈，它不仅可以清晰地展示你在任务中的表现，还可以为你的职业发展打下坚实的基础。以下是一个可供参考的中文大纲：
> 
> I. 引言
> II. 工作内容概述
>    A. 编写和修改公司内部的各种算法、数据结构及相关代码；
>    B. 解决团队日常开发过程中遇到的问题；
>    C. 协助产品经理设计并实现新功能；
>    D. 优化已有功能，提升应用性能；
>    E. 跟进项目进度，meeting 与 stakeholder，写报告等。
> 
> III. 自我评价
>    A. 描述你对自己工作的认识和理解；
>    B. 分析你在任务中的表现和达成的效果；
>    C. 提出你对未来工作的规划和目标。
> 
> IV. 结论
>    A. 总结全文，重申你对工作的贡献和明确你在未来工作中的方向；
>    B. 提出针对自己工作的建议和改进意见。

**Chinese-llama2-alpaca-7b**
> 1. 算法工程师的职责与要求
> 2. 常见的算法问题及其解决方案
> 3. 如何提高算法效率
> 4. 实际案例分析
> 5.  conclusion 1. 算法工程师的职责与要求
>    - 设计和开发公司内部的算法产品
>    - 理解并精通至少一种编程语言
>    - 掌握数据结构和算法的基本知识
> - 常见的算法问题及其解决方案
>    - 排序算法：快速排序、归并排序等
>    - 查找算法：哈希表、树搜索等
>    - 优化算法：分治策略、动态规划等
> - 如何提高算法效率
>    - 理解并精通更多的数据结构和算法
>    - 学习并熟练使用Python或其他编程语言
>    - 阅读和理解经典的算法书籍和论文
> - 实际案例分析
>    - 选择一到两个具有挑战性的算法问题进行深入研究
>    - 对所选案例进行详细描述和分析
> - conclusion 算法工程师的工作总结

**chinese-llama-2-7b**
> The work summary of algorithm engineer should be how to write better, help me write a Chinese outline and some attention points.

#### 4. 分析高考作文题

**prompt**
> 使用中文，根据给出的高考作文题目，进行以下分析任务：
> 
> 1、解读材料内容
> 2、分析出题人的考察思路
> 3、为考生提供写作建议。
> 
> 作文题目：
> 吹灭别人的灯，并不会让自己更加光明；阻挡别人的路，也不会让自己行得更远。“一花独放不是春，百花齐放春满园。”如果世界上只有一种花朵，就算这种花朵再美，那也是单调的。以上两则材料以生动形象的语言说出了普遍的道理。请据此写一篇文章，体现你的认识与思考。要求：选准角度，确定立意，明确文体，自拟标题；不要套作，不得抄袭；不得泄露个人信息；不少于800字。

**Chinese-llama2-CLAM-7b**
> 解读材料内容：
> 这个作文题目讲究的是综合素质和批判性思维。首先，题目要求我们吹灭别人的灯和阻挡别人的路，这是在引导我们从社会学或者哲学的视角来阐述问题。其次，“一花独放不是春，百花齐放春满园。”这句话直接引用了春天的象征意义，暗示了多元共存的重要性。
> 
> 分析出题人的考察思路：
> 题目要求我们从各种角度出发，深入理解和评价这个问题。同时，还强调了多样性和包容性的重要性，希望我们能够从中得到启发，写出具有深度和广度的文章。
> 
> 写作建议：
> 1. 可以从社会、历史、文化等多个层面出发，阐述吹灭别人的灯和阻挡别人的路的好处和弊端；
> 2. 可以结合自己的经验和观察，扩展讨论的范围；
> 3. 注意语言的生动性和逻辑的清晰性，保证文章的品质；
> 4. 尽量避免过于主观的表达，保持批判性和开放的思维。

**Chinese-llama2-alpaca-7b**
> 这次高考作文题目给考生带来的是一个广泛而又深刻的视角，要求考生从多元化的角度去理解和探索这个命题背后所包含的哲学智慧。首先，我们可以从物质层面解读这个题目，理解它寓含的实质：即达成互助互利，共同进步的人际关系应当是支持他人、尊重他人，并且愿意为他人照亮前方的路。其次，我们还可以从主观层面去思考，如何在日常生活之中践行这一理念，以及这样的做法对我们个人成长的积极影响。总的来说，这个作文题目旨在通过强制性的思考与实践，促使我们逐渐超越自我，变得更加开放与合作，从而真正理解并熟练运用这个题目所要传达的价值观。

**chinese-llama-2-7b**
> 注：本次考试采用全国统一命题和阅卷方式。### Response:题目：《吹熄别人的灯》
> 
> 解读材料内容：
> 第一则材料讲述了一个故事，主人公在夜晚遇到了一位需要帮助的陌生人，他没有拒绝，而是伸出援手，最终成功地帮助了对方。这个故事告诉我们，即使别人的困难可能会影响到我们自己，但是我们应该尽力去帮助他们，因为这样做不仅可以让别人感激我们，还可以增强彼此之间的友谊和互相支持的精神。
> 
> 第二则材料表达了一个观点，即多元化的环境可以带来更好的效益。它指出，如果每个人都只关注自己的利益，那么整个社会将变得单调乏味。相反，如果大家共同努力，尊重彼此的差异，那么整个社会将变得丰富多彩。
> 
> 分析出题人的考察思路：
> 这两则材料都涉及到了人际交往和团队合作的问题。首先，考生需要从材料中找到主旨句，即第一则材料的主旨句是“当我们帮助别人时，我们也在帮助自己”，第二则材料的主旨句是“多元化的环境可以带来更好的效益”。其次，考生需要深入理解这些主旨句所传递的含义，即第一则材料告诉我们，通过帮助别人，我们可以获得更多的朋友和支持，而第二则材料则强调了多元化的环境可以促进创新和发展。最后，考生需要运用这些主旨句来构建自己的文章，并且在文章中表达自己的想法和观点。
> 
> 为考生提供写作建议：
> 针对这两则材料，考生可以从以下几个方面着手：
> 
> 1. 探讨人际交往的重要性。考生可以谈论自己或身边的例子，阐述人际交往对于个人成长和幸福的重要性。
> 
> 2. 探讨团队合作的优势。考生可以列举实际案例，介绍团队合作的好处，比如协作、沟通、学习等。
> 
> 3. 探讨多元化的价值。考生可以引用第二则材料中的观点，阐述多元化的环境可以带来哪些好处，比如创新、发展、包容等。
> 
> 4. 结合自己的经历和观点，撰写一篇具有逻辑性和连贯性的文章。考生可以按照文章的结构，逐步展开自己的观点，并且通过实例和数据来支撑自己的观点。

#### 5. 中餐菜谱

**prompt**
> 麻婆豆腐是哪里的菜？写一份菜谱。

**Chinese-llama2-CLAM-7b**
> 麻婆豆腐是四川省的传统名菜，具有鲜香味美、麻辣鲜香、口感丰富等特点。下面是一份标准的麻婆豆腐菜谱：
> 
> 原料：嫩豆腐400克，牛肉末150克，葱、姜、蒜适量，豆瓣酱3大勺，郫县豆瓣酱2大勺，花椒粉1/2茶匙，生抽适量，老抽适量，白糖1大勺，食用油适量。
> 
> 步骤：
> 1. 豆腐切块，放开水中焯透备用。
> 2. 锅烧热，倒入适量油，放入牛肉末翻炒至变色。
> 3. 加入葱、姜、蒜爆炒出香味。
> 4. 接着加入豆瓣酱、郫县豆瓣酱，花椒粉，生抽，老抽，白糖搅拌均匀。
> 5. 倒入足够的清水，煮沸后加入焯透的豆腐，小火慢炖15分钟。
> 6. 待汤汁浓稠，汤色紧实时，关火撒上葱花即可。
> 
> 这是一道非常经典的川菜，口感独特，希望你能尝到满满的幸福感！

**Chinese-llama2-alpaca-7b**
> 麻婆豆腐是四川省成都市的名菜。它由嫩豆腐、肉末和辣椒面等食材制作而成，口感麻辣鲜香，色泽红亮。下面是简单的麻婆豆腐菜谱：
> 所需食材：
> 1.豆腐（适量）
> 2.瘦肉末（适量）
> 3.豆瓣酱（适量）
> 4.郫县豆瓣酱（适量）
> 5.花椒粉（适量）
> 6.生抽（适量）
> 7.盐（适量）
> 8.白糖（适量）
> 9.葱姜蒜（适量）
> 步骤：
> 1.把豆腐切块，放开水里焯透备用。
> 2.热锅冷油，加入葱姜蒜炒出香味，再加入瘦肉末翻煎至变色。
> 3.加入豆瓣酱、郫县豆瓣酱继续翻炒，然后加入料酒、生抽、白糖、适量的盐进行调味。
> 4.接着加入清水或者高汤，大火烧开后转小火慢慢熬煮，期间需不断用勺子撇去浮沫。
> 5.当汤汁快要收干时，加入焯透的豆腐块，大火迅速翻煮均匀，最后撒上花椒粉即可出锅。
> 这就是传统的麻婆豆腐，尽情享受它带给你的麻辣美味吧！

**chinese-llama-2-7b**
> Where is the dish "Ma Po Tofu"?

### 小结

在调试和对比不同模型输出的过程中，我们发现：

1. 从对比样例可以看出，当我们直接用自然语言描述任务，基线模型相比微调模型更容易“崩坏”。
2. 开源数据微调的模型似乎容易先续写几句prompt的问题，然后才开始进行回答。
3. 测试时发现指令结尾有没有换行符、有几个换行符，也会导致一定的输出变化。使用模型时如果遇到自动续写问题的情况，可以考虑在问题末尾多添加一个换行符。
4. 基于我们目前的观察来看，自研数据微调后的模型确实可以使模型更好地理解指令意图，生成更加恰当的回答。
5. 可以发现，训练至这一阶段的语言模型，在进行自由问答时，比底座模型有明显进步，生成答案的随机性还是较大，但无法保证优质答案的复现。


## 协议

目前开源的微调模型，遵循底座模型 [Chinese-LLaMA-2-7B](https://github.com/CVI-SZU/Linly) 和 [Llama 2](https://ai.meta.com/llama/license/) 对其相关模型的协议说明。
