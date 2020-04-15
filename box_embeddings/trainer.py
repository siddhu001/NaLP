import numpy
import time
import evaluate
import torch
import kb
import utils
import os


class Trainer(object):
    def __init__(self, scoring_function, regularizer, loss, optim, train, valid, test, verbose=0, batch_size=1000,
                 hooks=None, eval_batch=100, negative_count=10, gradient_clip=None, regularization_coefficient=0.01,
                 save_dir="./logs"):
        super(Trainer, self).__init__()
        self.scoring_function = scoring_function
        self.loss = loss
        self.regularizer = regularizer
        self.train = train
        self.test = test
        self.valid = valid
        self.optim = optim
        self.batch_size = batch_size
        self.negative_count = negative_count
        self.ranker = evaluate.ranker(self.scoring_function, kb.union([train, valid, test]))
        self.eval_batch = eval_batch
        self.gradient_clip = gradient_clip
        self.regularization_coefficient = regularization_coefficient
        self.save_directory = save_dir
        self.best_mrr_on_valid = {"valid_m": {"mrr": 0.0}, "test_m": {"mrr": 0.0}}
        self.verbose = verbose
        self.hooks = hooks if hooks else []

    def step(self, begin, end):
        facts = self.train.facts[begin:end, :]
        s = torch.autograd.Variable(torch.from_numpy(facts[:, 0])).cuda()
        r = torch.autograd.Variable(torch.from_numpy(facts[:, 1])).cuda()
        o = torch.autograd.Variable(torch.from_numpy(facts[:, 2])).cuda()
        if self.regularization_coefficient is not None:
            reg = self.regularizer(s, r, o)
            reg = reg / self.batch_size
        else:
            reg = 0
        if self.negative_count == 0:
            no = None
            fno = self.scoring_function(s, r, no)
            loss = self.loss(o, fno)
        else:
            no = torch.randint(0, len(self.train.entity_map), (end - begin, self.negative_count)).cuda()
            fp = self.scoring_function(s, r, o)
            fno = self.scoring_function(s, r, no)
            loss = self.loss(fp.unsqueeze(1), fno)

        loss += self.regularization_coefficient * reg
        x = loss.item()
        rg = reg.item()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        debug = ""
        if "post_epoch" in dir(self.scoring_function):
            debug = self.scoring_function.post_epoch()
        return x, rg, debug

    def save_state(self, mini_batches, valid_score, test_score):
        state = dict()
        state['mini_batches'] = mini_batches
        state['epoch'] = mini_batches * self.batch_size / self.train.facts.shape[0]
        state['model_name'] = type(self.scoring_function).__name__
        state['model_weights'] = self.scoring_function.state_dict()
        state['optimizer_state'] = self.optim.state_dict()
        state['optimizer_name'] = type(self.optim).__name__
        state['valid_score_m'] = valid_score['m']
        state['test_score_m'] = test_score['m']
        filename = os.path.join(self.save_directory, "epoch_%.1f_val_%5.2f_test_%5.2f.pt" % (state['epoch'],
                                                                                             state['valid_score_m'][
                                                                                                 'mrr'],
                                                                                             state['test_score_m'][
                                                                                                 'mrr']))

        # torch.save(state, filename)
        try:
            if (state['valid_score_m']['mrr'] >= self.best_mrr_on_valid["valid_m"]["mrr"]):
                print("Best Model details:\n", "valid_m", str(state['valid_score_m']), "test_m",
                      str(state["test_score_m"]))
                best_name = os.path.join(self.save_directory, "best_valid_model.pt")
                self.best_mrr_on_valid = {"valid_m": state['valid_score_m'], "test_m": state["test_score_m"]}
                if (os.path.exists(best_name)):
                    os.remove(best_name)
                torch.save(state, best_name)  # os.symlink(os.path.realpath(filename), best_name)
        except:
            utils.colored_print("red", "unable to save model")

    def load_state(self, state_file):
        state = torch.load(state_file, map_location='cpu')
        if state['model_name'] != type(self.scoring_function).__name__:
            utils.colored_print('yellow', 'model name in saved file %s is different from the name of current model %s' %
                                (state['model_name'], type(self.scoring_function).__name__))

        self.scoring_function.load_state_dict(state['model_weights'])
        if state['optimizer_name'] != type(self.optim).__name__:
            utils.colored_print('yellow', ('optimizer name in saved file %s is different from the name of current ' +
                                           'optimizer %s') %
                                (state['optimizer_name'], type(self.optim).__name__))
        self.optim.load_state_dict(state['optimizer_state'])
        return state['mini_batches']

    def start(self, max_epochs=50, batch_count=(20, 10), mb_start=0,current_learning_rate=0.001):
        start = time.time()
        losses = []
        count = 0
        mini_batch_count = 0
        #valid_score = evaluate.evaluate("valid ", self.ranker, self.valid, self.eval_batch, verbose=self.verbose, hooks=self.hooks)
        #test_score = evaluate.evaluate("test ", self.ranker, self.test, self.eval_batch, verbose=self.verbose, hooks=self.hooks)
        warm_up_steps = max_epochs // 2
        for i in range(mb_start, max_epochs):
            self.train.facts = self.train.facts[torch.randperm(self.train.facts.shape[0]), :]
            begin = 0
            while begin < self.train.facts.shape[0]:
                end = min(begin + self.batch_size, self.train.facts.shape[0])
                mini_batch_count += 1
                #end = begin + self.batch_size
                # valid_score = evaluate.evaluate("valid ", self.ranker, self.valid, self.eval_batch, verbose=self.verbose, hooks=self.hooks)
                # test_score = evaluate.evaluate("test ", self.ranker, self.test, self.eval_batch, verbose=self.verbose, hooks=self.hooks)
                l, reg, debug = self.step(begin, end)
                losses.append(l)
                suffix = ("| Current Loss %8.4f | " % l) if len(losses) != batch_count[0] else "| Average Loss %8.4f | " % \
                                                                                               (numpy.mean(losses))
                suffix += "reg %6.3f | time %6.0f ||" % (reg, time.time() - start)
                suffix += debug
                prefix = "Mini Batches %5d or %5.1f epochs" % (mini_batch_count, i + begin/self.train.facts.shape[0])
                utils.print_progress_bar(len(losses), batch_count[0], prefix=prefix, suffix=suffix)

                if len(losses) >= batch_count[0]:
                    # if len(losses) >= 0:
                    losses = []
                    count += 1
                    if count == batch_count[1]:
                        self.scoring_function.eval()
                        # train_score = evaluate.evaluate("train ", self.ranker, self.train.kb, self.eval_batch, verbose=self.verbose, hooks=self.hooks)
                        # valid_score = evaluate.evaluate("validn", self.ranker, self.valid, self.eval_batch,
                        #                                verbose=self.verbose, hooks=self.hooks)
                        test_score = evaluate.evaluate("test ", self.ranker, self.test, self.eval_batch,
                                                       verbose=self.verbose, hooks=self.hooks)
                        valid_score=test_score
                        self.scoring_function.train()
                        count = 0
                        print()
                        self.save_state(i, valid_score, test_score)

                begin += self.batch_size
            #if i >= warm_up_steps:
                #current_learning_rate = current_learning_rate / 10
                #self.optim = torch.optim.Adam(
                #    self.scoring_function.parameters(),
                #    lr=current_learning_rate
                #)
                #warm_up_steps = warm_up_steps * 3
        print()
        print("Ending")
        print(self.best_mrr_on_valid["valid_m"])
        print(self.best_mrr_on_valid["test_m"])

