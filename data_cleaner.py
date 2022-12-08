import xml.etree.ElementTree as ET
from typing import List, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from lxml import etree # type: ignore

class StackOverflowPost:
    def __init__(self, xml_row: ET.Element) -> None:
        attributes = xml_row.attrib
        self.id: int = int(attributes["Id"])
        self.post_type_id: int = int(attributes["PostTypeId"])
        self.accepted_answer_id: Optional[int] = (
            int(attributes["AcceptedAnswerId"])
            if "AcceptedAnswerId" in attributes
            else None
        )
        self.parent_id: Optional[int] = (
            int(attributes["ParentId"]) if "ParentId" in attributes else None
        )
        self.creation_date: str = attributes["CreationDate"]
        self.score: int = int(attributes["Score"])
        self.view_count: Optional[int] = (
            int(attributes["ViewCount"]) if "ViewCount" in attributes else None
        )
        self.body: str = attributes["Body"]
        self.owner_user_id: Optional[int] = (
            int(attributes["OwnerUserId"]) if "OwnerUserId" in attributes else None
        )
        self.owner_display_name: Optional[str] = (
            attributes["OwnerDisplayName"] if "OwnerDisplayName" in attributes else None
        )
        self.last_editor_user_id: Optional[int] = (
            int(attributes["LastEditorUserId"])
            if "LastEditorUserId" in attributes
            else None
        )
        self.last_editor_display_name: Optional[str] = (
            attributes["LastEditorDisplayName"]
            if "LastEditorDisplayName" in attributes
            else None
        )
        self.last_edit_date: Optional[str] = (
            attributes["LastEditDate"] if "LastEditDate" in attributes else None
        )
        self.last_activity_date: str = attributes["LastActivityDate"]
        self.title: Optional[str] = (
            attributes["Title"] if "Title" in attributes else None
        )
        tags: Optional[str] = attributes["Tags"] if "Tags" in attributes else None
        self.tags: list[str] = []
        if tags is not None:
            split_tags = tags.replace("<", " ").replace(">", " ").split()
            self.tags = [tag for tag in split_tags if tag != ""]

        self.answer_count: Optional[int] = (
            int(attributes["AnswerCount"]) if "AnswerCount" in attributes else None
        )
        self.comment_count: Optional[int] = (
            int(attributes["CommentCount"]) if "CommentCount" in attributes else None
        )
        self.favorite_count: Optional[int] = (
            int(attributes["FavoriteCount"]) if "FavoriteCount" in attributes else None
        )
        self.closed_date: Optional[str] = (
            attributes["ClosedDate"] if "ClosedDate" in attributes else None
        )
        self.community_owned_date: Optional[str] = (
            attributes["CommunityOwnedDate"]
            if "CommunityOwnedDate" in attributes
            else None
        )
        self.content_license: str = attributes["ContentLicense"]

        soup = BeautifulSoup(self.body, "html.parser")

        # Count the number of code snippets
        self.num_code_snippets = len(soup.find_all("code"))

        # Count the total length of all code snippets
        self.total_code_length = sum(len(code.text) for code in soup.find_all("code"))

        # Count the number of images
        self.num_images = len(soup.find_all("img"))

        # Get the title length
        self.title_length = len(self.title) if self.title is not None else 0

        # Count the number of tags
        self.num_tags = len(self.tags) if self.tags is not None else 0

        # Check if the post is answered
        self.is_answered = self.accepted_answer_id is not None

        # Calculate the text word count
        text = soup.get_text()
        self.text_word_count = len(text.split())

    def to_tensor_flow_input(self) -> list[int]:
        return [
            self.title_length,
            self.text_word_count,
            self.num_code_snippets,
            self.total_code_length,
            self.num_images,
            self.num_tags,
        ]

    def to_tensor_flow_output(self) -> bool:
        return self.is_answered

    def print_metadata(self):
        print(f"Link: https://stackoverflow.com/q/{self.id}")
        print(f"Title Length: {self.title_length}")
        print(f"Total Word Count: {self.text_word_count}")
        print(f"Number of Code Snippets: {self.num_code_snippets}")
        print(f"Total Code Length: {self.total_code_length}")
        print(f"Number of Images: {self.num_images}")
        print(f"Number of Tags: {self.num_tags}")


def parsePosts(files: list[str]) -> list[StackOverflowPost]:
    parsed_posts: list[StackOverflowPost] = []

    # Use a ThreadPoolExecutor to process the files in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit a task to parse each file
        tasks = [executor.submit(parseFile, file) for file in files]        
        # Iterate over the completed tasks and add their results to the parsed_posts deque
        for task in as_completed(tasks):
            try:
                parsed_posts.extend(task.result())
            except:
                print(f"Error parsing file {task}")

    return parsed_posts

def parseFile(file: str) -> list[StackOverflowPost]:
    print(f"Parsing {file}...")
    # Parse the XML file using lxml
    parsed_posts: list[StackOverflowPost] = []
    try:
        context = etree.iterparse(file, events=('start', 'end'))

        # Track the number of parsed posts
        num_parsed_posts = 0

        # Iterate over the file in a loop
        for event, xml_row in context:
            if event == 'end' and xml_row.tag == "row":
                try:
                    parsed_posts.append(StackOverflowPost(xml_row))
                    num_parsed_posts += 1

                except Exception as e:
                    print("Error parsing row", e)
                    pass

        return parsed_posts
    except Exception as e:
        print(f"Error parsing {file}", e)
        return parsed_posts

def parseChunkedPosts() -> list[StackOverflowPost]:
    files = [chr(i) for i in range(ord('a'), ord('p') + 1)]
    posts =  parsePosts([f"data/dataset/xa{ltr}" for ltr in files], None)
    print('Parsed', len(posts))
    unsolved_posts = 0
    solved_posts = 0
    for post in posts:
        if post.is_answered:
            solved_posts += 1
        else:
            unsolved_posts += 1
    print('Solved', solved_posts)
    print('Unsolved', unsolved_posts)
    print('Solved to unsolved ratio', solved_posts / unsolved_posts)
    return posts