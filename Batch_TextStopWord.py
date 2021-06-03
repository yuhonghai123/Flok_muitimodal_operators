# encoding=utf-8
import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


class Batch_TextStopWord(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        if params["type"] == "CHN":
            stop_words_str = ',\n?\n、\n。\n“\n”\n《\n》\n！\n，\n：\n；\n？\n人民\n末##末\n啊\n阿\n哎\n哎呀\n哎哟\n唉\n俺\n俺们\n按\n按照\n吧\n吧哒\n把\n罢了\n被\n本\n本着\n比\n比方\n比如\n鄙人\n彼\n彼此\n边\n别\n别的\n别说\n并\n并且\n不比\n不成\n不单\n不但\n不独\n不管\n不光\n不过\n不仅\n不拘\n不论\n不怕\n不然\n不如\n不特\n不惟\n不问\n不 只\n朝\n朝着\n趁\n趁着\n乘\n冲\n除\n除此之外\n除非\n除了\n此\n此间\n此外\n从\n从而\n打\n待\n但\n但是\n当\n当着\n到\n得\n的\n的话\n等\n等等\n地\n第\n叮咚\n对\n对于\n多\n多少\n而\n而况\n而且\n而是\n而外\n而言\n而已\n尔后\n反过来\n反过来说\n反之\n非但\n非徒\n否则\n嘎\n嘎登\n该\n赶\n个\n各\n各个\n各位\n各种\n各自\n给\n根据\n跟\n故\n故此\n固然\n关于\n管\n归\n果然\n果真\n过\n哈\n哈哈\n呵\n和\n何\n何处\n何况\n何时\n嘿\n哼\n哼唷\n呼哧\n乎\n哗\n还是\n还有\n换句话说\n换言之\n 或\n或是\n或者\n极了\n及\n及其\n及至\n即\n即便\n即或\n即令\n即若\n即使\n几\n几时\n己\n既\n既然\n既是\n继而\n加之\n假如\n假若\n假使\n鉴于\n将\n较\n较之\n叫\n 接着\n结果\n借\n紧接着\n进而\n尽\n尽管\n经\n经过\n就\n就是\n就是说\n据\n具体地说\n具体说来\n开始\n开外\n靠\n咳\n可\n可见\n可是\n可以\n况且\n啦\n来\n来着\n离\n例如\n哩\n连\n连同\n两者\n了\n临\n另\n另外\n另一方面\n论\n嘛\n吗\n慢说\n漫说\n冒\n么\n每\n每当\n们\n莫若\n某\n某个\n某些\n拿\n哪\n哪边\n哪儿\n哪个\n哪里\n哪年\n哪怕\n哪天\n哪些\n哪样\n那\n那边\n那儿\n那个\n那会儿\n那里\n那么\n那么些\n那么样\n那时\n那些\n那样\n乃\n乃至\n呢\n能\n你\n你们\n您\n宁\n宁可\n宁肯\n宁愿\n哦\n呕\n啪达\n旁人\n呸\n凭\n凭借\n其\n其次\n其二\n其他\n其它\n其一\n其余\n其中\n起\n起见\n岂但\n恰恰相反\n前后\n前者\n且\n然而\n然后\n然则\n让\n人家\n任\n任何\n任凭\n如\n如此\n如果\n如何\n如其\n如若\n如上所述\n若\n若非\n若是\n啥\n上下\n尚且\n设若\n设使\n甚而\n甚么\n甚至\n省得\n时候\n什么\n什么样\n使得\n是\n是的\n首先\n谁\n谁知\n顺\n顺着\n似的\n虽\n虽然\n虽说\n虽则\n随\n随着\n所\n所以\n他\n他们\n他人\n它\n它们\n她\n她们\n倘\n倘或\n倘然\n倘若\n倘使\n腾\n替\n通过\n同\n同时\n哇\n万一\n往\n望\n为\n为何\n为了\n为什么\n为着\n喂\n嗡嗡\n我\n我们\n呜\n呜呼\n乌乎\n无论\n无宁\n毋宁\n嘻\n吓\n相对而言\n像\n向\n向着\n嘘\n呀\n焉\n沿\n 沿着\n要\n要不\n要不然\n要不是\n要么\n要是\n也\n也罢\n也好\n一\n一般\n一旦\n一方面\n一来\n一切\n一样\n一则\n依\n依照\n矣\n以\n以便\n以及\n以免\n以至\n以至于\n以致\n抑或\n因\n因此\n因而\n因为\n哟\n用\n由\n由此可见\n由于\n有\n有的\n有关\n有些\n又\n于\n于是\n于是乎\n与\n与此同时\n与否\n与其\n越是\n云云\n哉\n再说\n再者\n在\n在下\n咱\n咱们\n则\n怎\n怎么\n怎么办\n怎么样\n怎样\n咋\n照\n照着\n者\n这\n这边\n这儿\n这个\n这会儿\n这就是说\n这里\n这么\n这么点儿\n这么些\n这么样\n 这时\n这些\n这样\n正如\n吱\n之\n之类\n之所以\n之一\n只是\n只限\n只要\n只有\n至\n至于\n诸位\n着\n着呢\n自\n自从\n自个儿\n自各儿\n自己\n自家\n自身\n综上所述\n 总的来看\n总的来说\n总的说来\n总而言之\n总之\n纵\n纵令\n纵然\n纵使\n遵照\n作为\n兮\n呃\n呗\n咚\n咦\n喏\n啐\n喔唷\n嗬\n嗯\n嗳\n~\n!\n.\n:\n"\n\'\n(\n)\n*\nA\n白\n社会主义\n--\n..\n>>\n [\n ]\n\n<\n>\n/\n\\\n|\n-\n_\n+\n=\n&\n^\n%\n#\n@\n`\n;\n$\n（\n）\n——\n—\n￥\n·\n...\n‘\n’\n〉\n〈\n…\n\u3000\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n０\n１\n２\n３\n４\n５\n６\n７\n８\n９\n二\n三\n四\n五\n六\n七\n八\n九\n零\n＞\n＜\n＠\n＃\n＄\n％\n︿\n＆\n＊\n＋\n～\n｜\n［\n］\n｛\n ｝\n啊哈\n啊呀\n啊哟\n挨次\n挨个\n挨家挨户\n挨门挨户\n挨门逐户\n挨着\n按理\n按期\n按时\n按说\n暗地里\n暗中\n暗自\n昂然\n八成\n白白\n半\n梆\n保管\n保险\n饱\n 背地里\n背靠背\n倍感\n倍加\n本人\n本身\n甭\n比起\n比如说\n比照\n毕竟\n必\n必定\n必将\n必须\n便\n别人\n并非\n并肩\n并没\n并没有\n并排\n并无\n勃然\n不\n不必\n 不常\n不大\n不但...而且\n不得\n不得不\n不得了\n不得已\n不迭\n不定\n不对\n不妨\n不管怎样\n不会\n不仅...而且\n不仅仅\n不仅仅是\n不经意\n不可开交\n不可抗拒\n不 力\n不了\n不料\n不满\n不免\n不能不\n不起\n不巧\n不然的话\n不日\n不少\n不胜\n不时\n不是\n不同\n不能\n不要\n不外\n不外乎\n不下\n不限\n不消\n不已\n不亦乐乎\n不 由得\n不再\n不择手段\n不怎么\n不曾\n不知不觉\n不止\n不止一次\n不至于\n才\n才能\n策略地\n差不多\n差一点\n常\n常常\n常言道\n常言说\n常言说得好\n长此下去\n长话 短说\n长期以来\n长线\n敞开儿\n彻夜\n陈年\n趁便\n趁机\n趁热\n趁势\n趁早\n成年\n成年累月\n成心\n乘机\n乘胜\n乘势\n乘隙\n乘虚\n诚然\n迟早\n充分\n充其极\n充其量\n抽冷子\n臭\n初\n出\n出来\n出去\n除此\n除此而外\n除此以外\n除开\n除去\n除却\n除外\n处处\n川流不息\n传\n传说\n传闻\n串行\n纯\n纯粹\n此后\n此中\n次第\n匆匆\n从不\n从此\n从此以后\n从古到今\n从古至今\n从今以后\n从宽\n从来\n从轻\n从速\n从头\n从未\n从无到有\n从小\n从新\n从严\n从优\n从早到晚\n从中\n从重\n凑巧\n粗\n存心\n达旦\n打从\n打开天窗说亮话\n大\n大不了\n大大\n大抵\n大都\n大多\n大凡\n大概\n大家\n大举\n大略\n大面儿上\n大事\n大体\n大体上\n大约\n大张旗鼓\n大致\n呆呆地\n带\n殆\n待到\n单\n单纯\n单单\n但愿\n弹指之间\n当场\n当儿\n当即\n当口儿\n当然\n当庭\n当头\n当下\n当真\n当中\n倒不如\n倒不如说\n倒是\n到处\n到底\n到了儿\n到目前 为止\n到头\n到头来\n得起\n得天独厚\n的确\n等到\n叮当\n顶多\n定\n动不动\n动辄\n陡然\n都\n独\n独自\n断然\n顿时\n多次\n多多\n多多少少\n多多益善\n多亏\n多年来\n 多年前\n而后\n而论\n而又\n尔等\n二话不说\n二话没说\n反倒\n反倒是\n反而\n反手\n反之亦然\n反之则\n方\n方才\n方能\n放量\n非常\n非得\n分期\n分期分批\n分头\n奋勇\n愤然\n风雨无阻\n逢\n弗\n甫\n嘎嘎\n该当\n概\n赶快\n赶早不赶晚\n敢\n敢情\n敢于\n刚\n刚才\n刚好\n刚巧\n高低\n格外\n隔日\n隔夜\n个人\n各式\n更\n更加\n更进一步\n更为\n公然\n共\n共总\n够瞧的\n姑且\n古来\n故而\n故意\n固\n怪\n怪不得\n惯常\n光\n光是\n归根到底\n归根结底\n过于\n毫不\n毫无\n毫无保留地\n毫无例外\n好在\n何必\n何尝\n何妨\n何苦\n何乐而不为\n何须\n何止\n很\n很多\n很少\n轰然\n后来\n呼啦\n忽地\n忽然\n互\n互相\n哗啦\n话说\n还\n恍然\n会\n豁然\n活\n伙同\n或多或少\n或许\n基本\n基本上\n基于\n极\n极大\n极度\n极端\n极力\n极其\n极为\n急匆匆\n即将\n即刻\n即是说\n几度\n几番\n几乎\n几经\n既...又\n继之\n加上\n加以\n间或\n简而言之\n简言之\n简直\n见\n将才\n将近\n将要\n交口\n较比\n较为\n接连不断\n接下来\n皆可\n截然\n截至\n藉以\n借此\n借以\n届时\n仅\n仅仅\n谨\n进来\n进去\n近\n近几年来\n近来\n近年来\n尽管如此\n尽可能\n尽快\n尽量\n尽然\n尽如人意\n尽心竭力\n尽心尽力\n尽早\n精光\n经常\n竟\n竟然\n究竟\n就此\n就地\n就算\n居然\n局外\n举凡\n据称\n据此\n据实\n据说\n据我所知\n据悉\n具体来说\n决不\n决非\n绝\n绝不\n绝顶\n绝对\n绝非\n均\n喀\n看\n看来\n看起来\n看上去\n看样子\n可好\n可能\n恐怕\n快\n快要\n来不及\n 来得及\n来讲\n来看\n拦腰\n牢牢\n老\n老大\n老老实实\n老是\n累次\n累年\n理当\n理该\n理应\n历\n立\n立地\n立刻\n立马\n立时\n联袂\n连连\n连日\n连日来\n连声\n连袂\n临到\n另方面\n另行\n另一个\n路经\n屡\n屡次\n屡次三番\n屡屡\n缕缕\n率尔\n率然\n略\n略加\n略微\n略为\n论说\n马上\n蛮\n满\n没\n没有\n每逢\n每每\n每时每刻\n猛然\n猛然间\n莫\n莫不\n莫非\n莫如\n默默地\n默然\n呐\n那末\n奈\n难道\n难得\n难怪\n难说\n内\n年复一年\n凝神\n偶而\n偶尔\n怕\n砰\n碰巧\n譬如\n偏偏\n乒\n平素\n颇\n 迫于\n扑通\n其后\n其实\n奇\n齐\n起初\n起来\n起首\n起头\n起先\n岂\n岂非\n岂止\n迄\n恰逢\n恰好\n恰恰\n恰巧\n恰如\n恰似\n千\n千万\n千万千万\n切\n切不可\n切莫\n 切切\n切勿\n窃\n亲口\n亲身\n亲手\n亲眼\n亲自\n顷\n顷刻\n顷刻间\n顷刻之间\n请勿\n穷年累月\n取道\n去\n权时\n全都\n全力\n全年\n全然\n全身心\n然\n人人\n仍\n仍旧\n仍然\n日复一日\n日见\n日渐\n日益\n日臻\n如常\n如此等等\n如次\n如今\n如期\n如前所述\n如上\n如下\n汝\n三番两次\n三番五次\n三天两头\n瑟瑟\n沙沙\n上\n上来\n上去'
            stopwords = [line.rstrip() for line in stop_words_str.split('\n')]
            for text_name, text in text_dict.items():
                seg_list = jieba.cut(text)
                text_dict[text_name] = " ".join([d for d in seg_list if d not in stopwords])
        elif params["type"] == "ENG":
            stopwords = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
                         'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
                         'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all',
                         'once',
                         'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll',
                         'you',
                         'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been',
                         'will',
                         'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she',
                         'again', 'be',
                         'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most',
                         'yourself',
                         'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i',
                         'does', 'both',
                         'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom',
                         'wouldn',
                         'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no',
                         'about',
                         'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers',
                         'wasn',
                         'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during',
                         'which']
            for text_name, text in text_dict.items():
                word_tokens = word_tokenize(text)
                text_dict[text_name] = " ".join([w for w in word_tokens if w not in stopwords])
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result

if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    # all_info = json.load(f)
    # all_info = {
    #     "input": ["data/chinese.txt"],
    #     "inputFormat":["txt"],
    #     "inputLocation": ["local_fs"],
    #     "output": ["data/chinese_stopword.txt"],
    #     "outputFormat": ["txt"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"language":"CHN"}#CHN/ENG
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]

    algorithm = Batch_TextStopWord()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


