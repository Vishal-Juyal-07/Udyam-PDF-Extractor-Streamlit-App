import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_doctr_model():
    return ocr_predictor(pretrained=True)

ocr_model = load_doctr_model()

EXPECTED_FORMAT = """
Return the following fields strictly as valid JSON:

{
  "UDYAM_REGISTRATION_NUMBER": "",
  "ENTERPRISE_NAME": "",
  "ENTERPRISE_TYPE": "",
  "MAJOR_ACTIVITY": "",
  "SOCIAL_CATEGORY_OF_ENTREPRENEUR": "",
  "NAME_OF_UNIT": "",
  "OFFICIAL_ADDRESS_OF_ENTERPRISE": "",
  "DATE_OF_INCORPORATION_OR_REGISTRATION_OF_ENTERPRISE": "",
  "DATE_OF_COMMENCEMENT_OF_PRODUCTION_OR_BUSINESS": "",
  "NATIONAL_INDUSTRY_CLASSIFICATION_CODES": {
        "NIC_2_DIGIT": [],
        "NIC_4_DIGIT": [],
        "NIC_5_DIGIT": []
  },
  "DATE_OF_UDYAM_REGISTRATION": ""
}
"""
def extract_with_gpt(raw_text: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an OCR post-processor. Always return valid JSON only."},
            {"role": "user", "content": f"Raw text:\n{raw_text}\n\nExpected output format:\n{EXPECTED_FORMAT}"}
        ]
    )
    return response.choices[0].message.content


st.title("Udyam Registration PDF Extractor")

uploaded_file = st.file_uploader("Upload your Udyam Registration PDF", type=["pdf"])

if uploaded_file is not None:
    document = DocumentFile.from_pdf(uploaded_file.read())
    result = ocr_model(document)
    raw_text = "\n".join([word.value for page in result.pages for block in page.blocks for line in block.lines for word in line.words])
    raw_text=raw_text[200:]
    st.subheader("Extracted Raw Text")
    st.text_area("Raw Text", raw_text, height=200)
    import re
    import json

    def safe_json_parse(gpt_output: str):
        """
        Extract JSON object from GPT output safely.
        Handles cases where GPT adds text before/after JSON.
        """
        try:
            # Direct attempt
            return json.loads(gpt_output)
        except json.JSONDecodeError:
            # Try extracting JSON with regex
            match = re.search(r"\{.*\}", gpt_output, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    if st.button("Process with LLM"):
        with st.spinner("Extracting structured data..."):
            gpt_output = extract_with_gpt(raw_text)

            data = safe_json_parse(gpt_output)

            if data:
                st.subheader("Extracted Information")

                for key, value in data.items():
                    if key == "NATIONAL_INDUSTRY_CLASSIFICATION_CODES":
                        st.markdown("**NATIONAL INDUSTRY CLASSIFICATION CODES:**")
                        st.markdown(f"- **NIC 2-DIGIT:** {', '.join(value.get('NIC_2_DIGIT', [])) or '❌ NOT FOUND'}")
                        st.markdown(f"- **NIC 4-DIGIT:** {', '.join(value.get('NIC_4_DIGIT', [])) or '❌ NOT FOUND'}")
                        st.markdown(f"- **NIC 5-DIGIT:** {', '.join(value.get('NIC_5_DIGIT', [])) or '❌ NOT FOUND'}")
                    else:
                        st.markdown(f"**{key}:** {value if value else '❌ NOT FOUND'}")
