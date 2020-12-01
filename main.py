import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import *

from autocorrect import Speller
from nltk.stem.snowball import SnowballStemmer

spell_checker = Speller(lang='en')
stemmer = SnowballStemmer('english')


class Sentiment(Enum):
    POSITIVE = 'Positive'
    NEUTRAL = 'Neutral'
    NEGATIVE = 'Negative'


class Recommendation(Enum):
    BUY = 'Buy'
    NEUTRAL = 'Neutral'
    DO_NOT_BUY = 'Do Not Buy'


@dataclass(frozen=True)
class Review:
    title: str
    content: str
    sentiment: Optional[bool] = None
    score: Optional[int] = None


def loader(path: str) -> str:
    with open(path) as f:
        return f.read()


NEGATIVES = loader('./data/negative-words.txt').split('\n')
POSITIVES = loader('./data/positive-words.txt').split('\n')


def splitter(data: str) -> Iterator['Review']:
    while data:
        current_title = re.search(r'\[t\]', data)
        data = data[current_title.end():] if current_title else ''
        title: str

        title, data = data.split('\n', 1)

        next_title = re.search(r'\[t\]', data)
        content: str = data[:next_title.start()] if next_title else data

        yield Review(title=title, content=content)

        data = data[next_title.start():] if next_title else ''


def take_words(content: str) -> list[str]:
    content = re.sub(r'\[([+-][1-3])\]', '', content.replace('##', ''))
    return content.split()


def find_features(review: 'Review') -> Iterable:
    for word in take_words(review.content):
        correct_word = spell_checker(word)
        if correct_word in POSITIVES or stemmer.stem(correct_word) in POSITIVES:
            yield correct_word, +1
        elif correct_word in NEGATIVES or stemmer.stem(correct_word) in NEGATIVES:
            yield correct_word, -1


def compiler(state: List[Dict[str, int]], review: 'Review') -> List[Dict[str, int]]:
    features_in_content = find_features(review)
    comment: Dict[str, int] = defaultdict(int)
    for feature, score in features_in_content:
        comment[feature] += score

    new_state = [*state, comment]
    return new_state


def aggregate(overall: Dict[str, int], comment: Dict[str, int]) -> Dict[str, int]:
    for feature, score in comment.items():
        overall[feature] += int(score)
    return overall


def group_by_feature_name(features: List[Dict[str, int]]) -> Dict[str, int]:
    feature_list: Dict[str, int] = defaultdict(int)
    grouped: Dict[str, int] = reduce(aggregate, features, feature_list)
    return grouped


def main():
    reviews: Iterator['Review'] = splitter(loader('./data/CanonG3.txt'))
    state: List[Dict[str, int]] = []
    comments: List[Dict[str, int]] = reduce(compiler, reviews, state)

    sentiments: List[Sentiment] = []
    with open('./data/out.txt', 'w+') as out:
        for index, comment in enumerate(comments):
            score: int = sum(comment.values(), 0)
            sentiment: Sentiment = (
                Sentiment.POSITIVE if score > 0
                else Sentiment.NEGATIVE if score < 0
                else Sentiment.NEUTRAL
            )
            sentiments.append(sentiment)
            out.write('\n\t'.join([
                f'{index + 1}:',
                f'sentiment = {sentiment.value}',
                f'{score = }\n',
            ]))

        overall = group_by_feature_name(comments)
        all_scores = overall.values()

        overall_score: int = sum(all_scores, 0)
        recommendation: Recommendation = (
            Recommendation.BUY if overall_score > 0
            else Recommendation.DO_NOT_BUY if overall_score < 0
            else Recommendation.NEUTRAL
        )

        recommendation_score: float

        if recommendation == Recommendation.BUY:
            p = len(list(filter(lambda sense: sense == Sentiment.POSITIVE, sentiments)))
        elif recommendation == Recommendation.DO_NOT_BUY:
            p = len(list(filter(lambda sense: sense == Sentiment.NEGATIVE, sentiments)))
        else:
            p = len(sentiments)
        recommendation_score = p / len(sentiments)

        best_score: int = max(overall.values())
        best_features: List[str] = [k for k, v in overall.items() if v == best_score]
        best_out = '\t\n'.join(best_features)

        worst_score: int = min(overall.values())
        worst_features: List[str] = [k for k, v in overall.items() if v == worst_score]
        worst_out = '\n\t'.join(worst_features)

        out.write('\n\t\t'.join([
            f'\nOverall:',
            f'recommendation = {recommendation.value}',
            f'recommendation_score = %{recommendation_score * 100:.2f}',
            f'best_features: {best_out}',
            f'worst_features: {worst_out}',
        ]))


if __name__ == '__main__':
    main()
