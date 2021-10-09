from bert4keras.tokenizers import Tokenizer


class Config(object):
    def __init__(self):
        self.base_dir = '/data1/home/fzq/projects/MRC/data/'  # 代码存放基础路径
        # 存模型和读数据参数
        self.processed_data = '/data1/home/fzq/projects/MRC/data/datasets/'  # 处理后的数据路径

        self.warmup_proportion = 0.05
        self.pretrainning_model = 'nezha'
        self.over_sample = True

        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 5

        self.train_epoch = 5
        self.sequence_length = 512
        self.q_len = 20  # 问题长度
        self.answer_len = 60  # 答案长度

        self.learning_rate = 1e-4
        self.embed_learning_rate = 5e-5
        self.batch_size = 2#原始为4

        self.embed_trainable = True

        self.as_encoder = True

        # 用于CRF的标签

        if self.pretrainning_model == 'nezha':
            model = '/data1/home/fzq/projects/nlpclassification/data/model/nezha-cn-base/nezha-cn-base/'
        elif self.pretrainning_model == 'roberta':
            model = '/data1/home/fzq/projects/nlpclassification/data/model/RoBERTa_zh_L12_PyTorch/'
        else:
            model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_electra_base/'

        self.model_path = model
        self.bert_config_file = model + 'config.json'
        self.bert_file = model + 'pytorch_model.bin'
        self.vocab_file = model + 'vocab.txt'
        self.match_token = Tokenizer(self.vocab_file, do_lower_case=False)

        self.continue_training = False
        # 下接结构
        self.mid_struct = 'bilstm'  # bilstm,idcnn,rtransformer,tener
        self.num_layers = 1  # 下游层数
        # bilstm
        self.lstm_hidden = 256  # Bilstm隐藏层size
        # idcnn
        self.filters = 128  # idcnn
        self.kernel_size = 9
        # Tener
        self.num_layers = 1
        self.tener_hs = 256
        self.num_heads = 4
        # rTansformer
        self.k_size = 32
        self.rtrans_heads = 4

        self.save_model = self.base_dir + 'Savemodel/'
        self.fold = 1
        self.compare_result = False
        self.drop_prob = 0.1  # drop_out率
        # 卷积参数
        # self.gru_hidden_dim = 64
        self.rnn_num = 256
        self.dropout = 0.9

        self.early_stop = 100
        self.adv = ''
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        self.result_file="/data1/home/fzq/projects/MRC/data/result/"

        self.checkpoint_path = "/home/wangzhili/lei/Search_QA/Savemodel/runs_1,2/1610518337/model_0.4355_11462.bin"
        self.fuse_bert = 'dym'
        """
        实验记录
        """
