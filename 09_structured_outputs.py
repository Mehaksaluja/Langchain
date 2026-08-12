from pydantic import BaseModel, Field
from typing import Literal, Optional
from typing_extensions import TypedDict
import json


# =============================================================================
# SECTION 1 — What is a Pydantic model
# =============================================================================
# The core problem: model.invoke() returns free text — "The name is John Doe..."
# You cannot reliably parse that in production.
# Structured output forces the model to return an EXACT shape every time.
# Pydantic BaseModel is how you define that shape.

class ContactInfo(BaseModel):
    """Contact information extracted from text."""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address of the person")
    phone: str = Field(description="Phone number of the person")


def section_1_pydantic_basics():
    print("\n" + "="*60)
    print("SECTION 1 — Pydantic model basics")
    print("="*60)

    # Create an instance — same shape the model will return
    contact = ContactInfo(
        name="John Doe",
        email="john@example.com",
        phone="555-1234"
    )

    print("\n--- The object ---")
    print(contact)

    print("\n--- Access fields with dot notation ---")
    print("name :", contact.name)
    print("email:", contact.email)
    print("phone:", contact.phone)

    print("\n--- Convert to plain dict ---")
    print(contact.model_dump())

    print("\n--- Convert to JSON string ---")
    print(contact.model_dump_json())

    print("\n--- Schema that gets sent to the model ---")
    print(json.dumps(ContactInfo.model_json_schema(), indent=2))

    print("\n--- Validation: wrong type is blocked automatically ---")
    try:
        bad = ContactInfo(name="John", email=123, phone="555")
    except Exception as e:
        print(f"Caught {type(e).__name__} — email=123 (int) rejected, must be str")

    print("\nKEY INSIGHT:")
    print("  Free text response  -> you must parse strings (fragile, breaks in prod)")
    print("  Structured output   -> you get a Python object, access with .fieldname")
    print("  Field(description)  -> model reads this to know what to put in each field")
    print("  Validation          -> wrong types caught before reaching your app code")


# =============================================================================
# SECTION 2 — Richer schemas: Optional, Literal, list
# =============================================================================
# Real production schemas are more complex.
# Optional = field might not exist in the text
# Literal  = only specific values allowed
# list     = multiple items

class ProductReview(BaseModel):
    """Analysis of a product review."""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating: Optional[int] = Field(
        default=None,
        description="Numeric rating 1-5 if explicitly mentioned, else null"
    )
    key_points: list[str] = Field(
        description="List of main points from the review. Each point 2-4 words."
    )
    summary: str = Field(
        description="One sentence summary of the entire review"
    )


def section_2_richer_schemas():
    print("\n" + "="*60)
    print("SECTION 2 — Richer schemas: Optional, Literal, list")
    print("="*60)

    print("\n--- Schema ---")
    print(json.dumps(ProductReview.model_json_schema(), indent=2))

    print("\n--- Instance with all fields ---")
    review = ProductReview(
        sentiment="positive",
        rating=5,
        key_points=["fast shipping", "great quality", "good packaging"],
        summary="Excellent product with fast delivery."
    )
    print(review)
    print("key_points is a real list:", review.key_points)
    print("loop over it:", [p.upper() for p in review.key_points])

    print("\n--- Instance with Optional field missing ---")
    review2 = ProductReview(
        sentiment="neutral",
        rating=None,
        key_points=["average quality"],
        summary="Decent product, nothing special."
    )
    print(review2)
    print("rating is None:", review2.rating is None)

    print("\n--- Literal blocks invalid values ---")
    try:
        bad = ProductReview(
            sentiment="amazing",
            rating=5,
            key_points=[],
            summary="good"
        )
    except Exception as e:
        print(f"Caught {type(e).__name__} — 'amazing' not in allowed values")

    print("\nKEY INSIGHT:")
    print("  Optional[int]         -> field can be None (data might be missing)")
    print("  Literal['a','b','c']  -> only these exact values allowed")
    print("  list[str]             -> model returns multiple items as a Python list")


# =============================================================================
# SECTION 3 — TypedDict (simpler alternative to Pydantic)
# =============================================================================
# TypedDict = a plain typed dictionary, no validation
# Returns a plain Python dict (not an object with dot notation)

class MovieInfo(TypedDict):
    """Information about a movie."""
    title: str
    year: int
    genre: str
    director: str


def section_3_typeddict():
    print("\n" + "="*60)
    print("SECTION 3 — TypedDict (simpler alternative)")
    print("="*60)

    movie: MovieInfo = {
        "title": "Inception",
        "year": 2010,
        "genre": "sci-fi",
        "director": "Christopher Nolan"
    }

    print("TypedDict is just a dict:", movie)
    print("Access with []:", movie["title"])
    print("Access with .get():", movie.get("year"))

    print("\n--- Pydantic vs TypedDict ---")
    print("Pydantic:")
    print("  - Returns an object  -> result.name (dot notation)")
    print("  - Has validation     -> wrong type = error")
    print("  - Has Field()        -> per-field descriptions for model")
    print("  - Use for production -> stricter, safer")
    print()
    print("TypedDict:")
    print("  - Returns a dict     -> result['name'] (bracket notation)")
    print("  - No validation      -> wrong type passes silently")
    print("  - No Field()         -> no per-field descriptions")
    print("  - Use when           -> simple dict is enough, less boilerplate")

    print("\nKEY INSIGHT:")
    print("  In production, prefer Pydantic — catches model mistakes early")
    print("  TypedDict is fine for quick scripts or internal pipelines")


# =============================================================================
# SECTION 4 — with_structured_output() on the model directly
# NEEDS: OPENAI_API_KEY
# =============================================================================
# Wrap the model with a schema.
# Every .invoke() now returns that schema instead of free text.

def section_4_with_structured_output():
    from langchain.chat_models import init_chat_model

    print("\n" + "="*60)
    print("SECTION 4 — with_structured_output() (needs API key)")
    print("="*60)

    model = init_chat_model("gpt-4o-mini")

    # Wrap the model — now every invoke returns ContactInfo, not a string
    structured_model = model.with_structured_output(ContactInfo)

    print("\n--- Extract from messy text ---")
    result = structured_model.invoke(
        "Extract contact info: Hi I'm John Doe, "
        "reach me at john@example.com or call 555-1234"
    )

    print("type   :", type(result))       # ContactInfo — real Pydantic object
    print("result :", result)
    print("name   :", result.name)        # dot notation — no string parsing
    print("email  :", result.email)
    print("as dict:", result.model_dump())

    print("\n--- Extract from different format ---")
    result2 = structured_model.invoke(
        "Contact: Alice Smith | alice@company.org | +1-800-999-0001"
    )
    print(result2)

    print("\nKEY INSIGHT:")
    print("  model.with_structured_output(Schema) -> wraps the model")
    print("  Every .invoke() now returns Schema instance, not a string")
    print("  result.name works — no parsing, no regex, no splitting")


# =============================================================================
# SECTION 5 — Real use case: extract structured data from reviews
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_5_real_usecase_extraction():
    from langchain.chat_models import init_chat_model

    print("\n" + "="*60)
    print("SECTION 5 — Real use case: review extraction (needs API key)")
    print("="*60)

    model = init_chat_model("gpt-4o-mini")
    structured_model = model.with_structured_output(ProductReview)

    reviews = [
        "This product is absolutely amazing! 5 stars. Fast shipping, "
        "great build quality, and very reasonable price. Highly recommend.",

        "Terrible experience. Item arrived broken, customer service useless. "
        "1 out of 5. Complete waste of money.",

        "It's okay I guess. Does what it says but nothing special. "
        "Packaging was fine.",
    ]

    for i, text in enumerate(reviews, 1):
        print(f"\n--- Review {i} ---")
        print(f"Input : {text[:60]}...")
        result = structured_model.invoke(
            f"Analyze this product review:\n\n{text}"
        )
        print(f"sentiment  : {result.sentiment}")
        print(f"rating     : {result.rating}")
        print(f"key_points : {result.key_points}")
        print(f"summary    : {result.summary}")

    print("\nKEY INSIGHT:")
    print("  3 different reviews -> always the same predictable shape")
    print("  Now you can sort by sentiment, filter by rating, etc.")
    print("  This is the foundation of every document extraction pipeline")


# =============================================================================
# SECTION 6 — TypedDict with the model
# NEEDS: OPENAI_API_KEY
# =============================================================================

def section_6_typeddict_with_model():
    from langchain.chat_models import init_chat_model

    print("\n" + "="*60)
    print("SECTION 6 — TypedDict with model (needs API key)")
    print("="*60)

    model = init_chat_model("gpt-4o-mini")
    structured_model = model.with_structured_output(MovieInfo)

    result = structured_model.invoke(
        "Extract movie info: Inception (2010) is a sci-fi film "
        "directed by Christopher Nolan"
    )

    print("type   :", type(result))       # plain dict
    print("result :", result)
    print("title  :", result["title"])    # bracket notation
    print("year   :", result["year"])

    print("\nKEY INSIGHT:")
    print("  TypedDict -> plain dict -> use result['field']")
    print("  Pydantic  -> object     -> use result.field")
    print("  Both enforce the shape — pick based on what downstream code needs")


# =============================================================================
# SECTION 7 — include_raw: get parsed result AND raw message together
# NEEDS: OPENAI_API_KEY
# =============================================================================
# Sometimes you need the structured output AND the raw AIMessage
# (for token counts, logging, or debugging validation failures)

def section_7_include_raw():
    from langchain.chat_models import init_chat_model

    print("\n" + "="*60)
    print("SECTION 7 — include_raw=True (needs API key)")
    print("="*60)

    model = init_chat_model("gpt-4o-mini")

    # include_raw=True returns a dict with 3 keys:
    # 'raw'           -> the original AIMessage
    # 'parsed'        -> your structured object
    # 'parsing_error' -> None if ok, Exception if model output failed validation
    structured_model = model.with_structured_output(ContactInfo, include_raw=True)

    result = structured_model.invoke("Extract: Bob Smith, bob@test.com, 777-8888")

    print("Keys in result:", list(result.keys()))

    print("\n--- parsed: your structured object ---")
    print(result["parsed"])
    print("type:", type(result["parsed"]))

    print("\n--- raw: the original AIMessage ---")
    print("type  :", type(result["raw"]))
    print("usage :", result["raw"].usage_metadata)   # token counts

    print("\n--- parsing_error ---")
    print("parsing_error:", result["parsing_error"])  # None = all good

    print("\nKEY INSIGHT:")
    print("  Default (include_raw=False) -> just return the parsed object")
    print("  include_raw=True            -> {'raw', 'parsed', 'parsing_error'}")
    print("  Use when you need token counts, logging, or debugging")
    print("  parsing_error not None      -> model returned invalid output")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    section_1_pydantic_basics()
    section_2_richer_schemas()
    section_3_typeddict()
    # section_4_with_structured_output()
    # section_5_real_usecase_extraction()
    # section_6_typeddict_with_model()
    # section_7_include_raw()