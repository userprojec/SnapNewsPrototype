
import streamlit as st
import requests
import nltk
from newspaper import Article
from PIL import Image
from io import BytesIO
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Download NLTK punkt tokenizer data
nltk.download('punkt')


# Function to fetch multiple latest news articles from Google News API
def fetch_latest_articles(num_articles=1):
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "in",
        "apiKey": "2ebc5b505303468e85cd26cef759fb02",
        "pageSize": num_articles  # Specify the number of articles to fetch
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("articles", [])
    return articles


# Function to summarize article using LSA algorithm
def summarize_with_lsa(article_text):
    parser = PlaintextParser.from_string(article_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count=2)  # Limit summary to 2 sentences
    return " ".join(str(sentence) for sentence in summary)


# Main Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>SnapNews</h1><br>", unsafe_allow_html=True)

    articles = fetch_latest_articles(20)  # Fetch up to 20 latest articles
    if articles:
        for i, article in enumerate(articles, start=1):
            # st.write(f"### Article {i}")
            article_url = article['url']
            article_obj = Article(article_url)
            article_obj.download()
            article_obj.parse()

            # Paraphrase the title
            title = article['title']  # Paraphrase the title

            # Summarize the content using LSA algorithm
            summarized_content = summarize_with_lsa(article_obj.text)  # Summarize the content

            # Link to the original article source
            source_url = article['url']

            # Display the article content and summary
            st.write(f"**{title}**")

            # Create a two-column layout
            col1, col2 = st.columns([3, 1])

            # Display content summary in the left column
            with col1:
                st.write(f"{summarized_content}")
                st.write(f"**Published At:** {article['publishedAt']}")
                st.write(
                    f"**Source:** [{article['source']['name']}]({source_url})")  # Link to the original article source

            # Display the article image in the right column if available
            with col2:
                if 'urlToImage' in article:
                    image_url = article['urlToImage']
                    try:
                        image = Image.open(BytesIO(requests.get(image_url).content))
                        st.image(image, caption="Image Source: " + article['source']['name'], use_column_width=False, width=300)
                    except Exception as e:
                        pass  # If image loading fails, do nothing
                else:
                    pass  # If no image available, leave the column blank

            st.write("---")
    else:
        st.error("Failed to fetch the latest news articles.")


if __name__ == "__main__":
    main()
