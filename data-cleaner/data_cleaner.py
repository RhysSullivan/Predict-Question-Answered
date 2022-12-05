import xml.etree.ElementTree as ET
from typing import Optional
from bs4 import BeautifulSoup


class Posts:
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


# Parse the XML file
# context = ET.iterparse("data/Posts.xml")
context = ET.iterparse("data-cleaner/sample.xml")

# Iterate over the elements of the XML file
for event, xml_row in context:
    # Check if the element is a "row" element
    if xml_row.tag == "row":
        # Create a Posts object for each row
        post = Posts(xml_row)
        print(post.id, len(post.body), post.tags)
