from app.models import CandidateProfile, Project


candidate_profile = CandidateProfile(
    name="Ian Farai Madhara",
    target_roles=[
        "AI Engineer",
        "Backend Software Engineer",
        "Full Stack Software Engineer",
    ],
    skills=[
        "Python",
        "Java",
        "TypeScript",
        "FastAPI",
        "Spring Boot",
        "React",
        "PostgreSQL",
        "Docker",
    ],
    projects=[
        Project(
            id="stock-prediction",
            name="Stock Prediction App",
            description=(
                "A full-stack application for stock prediction "
                "using machine learning."
            ),
            technologies=[
                "Python",
                "FastAPI",
                "React",
                "Docker",
                "scikit-learn",
                "MLflow",
            ],
            evidence=[
                "Built a FastAPI backend and React frontend.",
                "Used RSI and rolling statistics as model features.",
                "Compared linear regression, SVM, and random forest models.",
                "Containerized the application with Docker.",
            ],
        ),
        Project(
            id="rag-tutor",
            name="RAG TutorBot",
            description=(
                "A tutoring application that retrieves relevant "
                "learning material to support responses."
            ),
            technologies=[
                "Python",
                "Next.js",
                "PostgreSQL",
                "pgvector",
            ],
            evidence=[
                "Converted notebooks into Markdown for ingestion.",
                "Stored document embeddings in PostgreSQL using pgvector.",
                "Built a Next.js chat interface.",
            ],
        ),
    ],
)