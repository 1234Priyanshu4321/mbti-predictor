import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved models
with open("models/ie_model.pkl", "rb") as f:
    ie_model = joblib.load(f)
with open("models/ns_model.pkl", "rb") as f:
    ns_model = joblib.load(f)
with open("models/tf_model.pkl", "rb") as f:
    tf_model = joblib.load(f)
with open("models/jp_model.pkl", "rb") as f:
    jp_model = joblib.load(f)
    
# Function to predict MBTI from individual predictions
def get_mbti(ie, ns, tf, jp):
    return (
        ("E" if ie else "I") +
        ("N" if ns else "S") +
        ("F" if tf else "T") +
        ("P" if jp else "J")
    )

# Function to show personality type explanation
def explain_personality_type(mbti):
    explanations = {
        "INTJ": """**INTJ – The Architect**

- **Strengths**: Strategic, independent, analytical, determined.
- **Weaknesses**: Arrogant, dismissive of emotions, perfectionist.
- **Ideal Careers**: Scientist, strategist, software engineer, architect.
- **Famous INTJs**: Elon Musk, Arnold Schwarzenegger, Mark Zuckerberg.""",

        "INTP": """**INTP – The Thinker**

- **Strengths**: Innovative, curious, logical, independent.
- **Weaknesses**: Overly analytical, absent-minded, socially withdrawn.
- **Ideal Careers**: Theorist, programmer, physicist, philosopher.
- **Famous INTPs**: Albert Einstein, Bill Gates, Tina Fey.""",

        "ENTJ": """**ENTJ – The Commander**

- **Strengths**: Efficient, bold, strong-willed, strategic.
- **Weaknesses**: Stubborn, intolerant, impatient with emotions.
- **Ideal Careers**: CEO, lawyer, executive, entrepreneur.
- **Famous ENTJs**: Steve Jobs, Gordon Ramsay, Margaret Thatcher.""",

        "ENTP": """**ENTP – The Debater**

- **Strengths**: Energetic, quick-witted, outgoing, confident.
- **Weaknesses**: Argumentative, insensitive, easily bored.
- **Ideal Careers**: Inventor, entrepreneur, journalist, lawyer.
- **Famous ENTPs**: Tom Hanks, Mark Twain, Thomas Edison.""",

        "INFJ": """**INFJ – The Advocate**

- **Strengths**: Insightful, compassionate, determined, creative.
- **Weaknesses**: Perfectionist, sensitive, can burn out easily.
- **Ideal Careers**: Counselor, writer, psychologist, social worker.
- **Famous INFJs**: Carl Jung, Martin Luther King Jr., Lady Gaga.""",

        "INFP": """**INFP – The Mediator**

- **Strengths**: Empathetic, idealistic, open-minded, creative.
- **Weaknesses**: Overly idealistic, self-critical, avoids conflict.
- **Ideal Careers**: Writer, artist, counselor, social activist.
- **Famous INFPs**: J.R.R. Tolkien, William Shakespeare, Alicia Keys.""",

        "ENFJ": """**ENFJ – The Protagonist**

- **Strengths**: Charismatic, altruistic, inspiring, organized.
- **Weaknesses**: Overly idealistic, sensitive, struggles with criticism.
- **Ideal Careers**: Teacher, coach, politician, social worker.
- **Famous ENFJs**: Barack Obama, Oprah Winfrey, Jennifer Lawrence.""",

        "ENFP": """**ENFP – The Campaigner**

- **Strengths**: Enthusiastic, creative, sociable, empathetic.
- **Weaknesses**: Overly emotional, easily distracted, dislikes routine.
- **Ideal Careers**: Actor, designer, psychologist, public speaker.
- **Famous ENFPs**: Robin Williams, Robert Downey Jr., Quentin Tarantino.""",

        "ISTJ": """**ISTJ – The Logistician**

- **Strengths**: Responsible, detail-oriented, dependable, loyal.
- **Weaknesses**: Rigid, judgmental, uncomfortable with change.
- **Ideal Careers**: Accountant, police officer, judge, military officer.
- **Famous ISTJs**: Angela Merkel, George Washington, Natalie Portman.""",

        "ISFJ": """**ISFJ – The Defender**

- **Strengths**: Warm, meticulous, loyal, responsible.
- **Weaknesses**: Shy, avoids conflict, overly humble.
- **Ideal Careers**: Nurse, librarian, teacher, caregiver.
- **Famous ISFJs**: Mother Teresa, Beyoncé, Kate Middleton.""",

        "ESTJ": """**ESTJ – The Executive**

- **Strengths**: Organized, practical, dedicated, strong leader.
- **Weaknesses**: Inflexible, insensitive, too focused on status.
- **Ideal Careers**: Manager, military leader, auditor, lawyer.
- **Famous ESTJs**: Michelle Obama, Judge Judy, Frank Sinatra.""",

        "ESFJ": """**ESFJ – The Consul**

- **Strengths**: Caring, social, loyal, cooperative.
- **Weaknesses**: Approval-seeking, sensitive to criticism.
- **Ideal Careers**: Nurse, event planner, teacher, HR manager.
- **Famous ESFJs**: Taylor Swift, Jennifer Garner, Bill Clinton.""",

        "ISTP": """**ISTP – The Virtuoso**

- **Strengths**: Practical, bold, hands-on problem solver.
- **Weaknesses**: Unemotional, impulsive, easily bored.
- **Ideal Careers**: Mechanic, engineer, athlete, pilot.
- **Famous ISTPs**: Clint Eastwood, Bruce Lee, Scarlett Johansson.""",

        "ISFP": """**ISFP – The Adventurer**

- **Strengths**: Artistic, sensitive, adaptable, curious.
- **Weaknesses**: Easily stressed, avoids confrontation.
- **Ideal Careers**: Artist, fashion designer, musician, vet.
- **Famous ISFPs**: Marilyn Monroe, Michael Jackson, Frida Kahlo.""",

        "ESTP": """**ESTP – The Entrepreneur**

- **Strengths**: Energetic, spontaneous, confident, action-oriented.
- **Weaknesses**: Impatient, risk-prone, may ignore rules.
- **Ideal Careers**: Sales, emergency services, entrepreneur.
- **Famous ESTPs**: Ernest Hemingway, Madonna, Bruce Willis.""",

        "ESFP": """**ESFP – The Entertainer**

- **Strengths**: Outgoing, friendly, spontaneous, fun-loving.
- **Weaknesses**: Easily bored, may struggle with planning.
- **Ideal Careers**: Performer, coach, motivational speaker.
- **Famous ESFPs**: Elvis Presley, Jamie Oliver, Adele."""
    }
    return explanations.get(mbti, "Personality type explanation not available.")

def get_advice(mbti):
    advice = {
        "INTJ": """• **Careers**: Excel in strategic, analytical roles like scientist, engineer, or architect.  
• **Strengths to Leverage**: Long‑term planning and independent problem‑solving.  
• **Dating**: Seek partners who respect your need for independence and intellectual depth.  
• **Tip**: Schedule deep, meaningful one-on-one conversations to build trust.""",

        "INTP": """• **Careers**: Thrive in research, academia, or software development.  
• **Strengths to Leverage**: Creative theorizing and objective analysis.  
• **Dating**: Look for someone curious who appreciates your quirky sense of humor.  
• **Tip**: Share your evolving ideas to foster engaging, meaningful dialogue.""",

        "ENTJ": """• **Careers**: Lead teams as a CEO, project manager, or executive.  
• **Strengths to Leverage**: Decisiveness, vision, and organizational prowess.  
• **Dating**: Partner with someone confident who values your ambition and drive.  
• **Tip**: Balance goal‑setting with emotional check‑ins to strengthen relationships.""",

        "ENTP": """• **Careers**: Innovate in startups, marketing, or law.  
• **Strengths to Leverage**: Quick thinking, debate skills, and adaptability.  
• **Dating**: Find someone open‑minded who can keep up with your energy and ideas.  
• **Tip**: Channel your debating skill into playful, respectful banter.""",

        "INFJ": """• **Careers**: Shine as a counselor, writer, or social advocate.  
• **Strengths to Leverage**: Empathy, vision, and deep insight into others.  
• **Dating**: Seek a partner who shares your values and supports your ideals.  
• **Tip**: Open up gradually—your depth can be as inviting as it is intense.""",

        "INFP": """• **Careers**: Flourish in creative writing, art, or counseling.  
• **Strengths to Leverage**: Idealism, empathy, and authenticity.  
• **Dating**: Connect with someone who appreciates your depth and imagination.  
• **Tip**: Share your inner world through creative expression to build intimacy.""",

        "ENFJ": """• **Careers**: Lead in teaching, coaching, or non‑profit management.  
• **Strengths to Leverage**: Charisma, empathy, and motivational skills.  
• **Dating**: Choose someone who values your nurturing and inspirational nature.  
• **Tip**: Balance giving with receiving—let your partner support you too.""",

        "ENFP": """• **Careers**: Thrive in PR, design, or entrepreneurship.  
• **Strengths to Leverage**: Enthusiasm, creativity, and people skills.  
• **Dating**: Partner with someone spontaneous who shares your zest for life.  
• **Tip**: Ground your excitement with shared goals to keep the spark sustainable.""",

        "ISTJ": """• **Careers**: Succeed in accounting, law enforcement, or administration.  
• **Strengths to Leverage**: Reliability, attention to detail, and structure.  
• **Dating**: Seek stability in a partner who respects routines and plans.  
• **Tip**: Show flexibility occasionally to deepen emotional connection.""",

        "ISFJ": """• **Careers**: Excel as a nurse, teacher, or caregiver.  
• **Strengths to Leverage**: Loyalty, warmth, and meticulous support.  
• **Dating**: Find someone who appreciates your nurturing and thoughtfulness.  
• **Tip**: Voice your own needs gently so they don’t go unnoticed.""",

        "ESTJ": """• **Careers**: Lead in management, law, or project coordination.  
• **Strengths to Leverage**: Organization, decisiveness, and practicality.  
• **Dating**: Seek a partner who values your leadership and dependability.  
• **Tip**: Practice active listening to show you care beyond tasks.""",

        "ESFJ": """• **Careers**: Shine in HR, event planning, or teaching.  
• **Strengths to Leverage**: Sociability, cooperation, and care for others.  
• **Dating**: Partner with someone who communicates openly and warmly.  
• **Tip**: Set boundaries kindly to maintain your own well‑being.""",

        "ISTP": """• **Careers**: Thrive in engineering, mechanics, or emergency services.  
• **Strengths to Leverage**: Practical problem‑solving and adaptability.  
• **Dating**: Look for a flexible partner who enjoys spontaneity and hands‑on fun.  
• **Tip**: Share your adventures and invite them into your projects.""",

        "ISFP": """• **Careers**: Excel in art, design, or animal care.  
• **Strengths to Leverage**: Creativity, sensitivity, and spontaneity.  
• **Dating**: Seek someone who honors your need for freedom and beauty.  
• **Tip**: Use creative activities together to express feelings without words.""",

        "ESTP": """• **Careers**: Succeed in sales, sports, or entrepreneurship.  
• **Strengths to Leverage**: Boldness, resourcefulness, and quick decision‑making.  
• **Dating**: Partner with someone who matches your energy and sense of adventure.  
• **Tip**: Slow down occasionally to deepen emotional bonds.""",

        "ESFP": """• **ESFP – The Entertainer**  
• **Strengths**: Outgoing, fun-loving, and spontaneous.  
• **Weaknesses**: Easily bored, may struggle with long-term plans.  
• **Dating**: Choose someone who enjoys living in the moment with you.  
• **Tip**: Plan meaningful surprises to keep your relationship vibrant."""
    }
    return advice.get(mbti, "Advice not available.")


# Streamlit app
st.set_page_config(page_title="MBTI Personality Quiz", layout="centered")
st.title("MBTI Personality Quiz")

# Session state
if "answers" not in st.session_state:
    st.session_state.answers = [5] * 15
if "page" not in st.session_state:
    st.session_state.page = 1

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# Page 1 – IE
if st.session_state.page == 1:
    st.subheader("Page 1: Introversion (I) vs Extraversion (E)")
    st.session_state.answers[0] = st.slider("1. I enjoy spending time with others.", 1, 10, st.session_state.answers[0])
    st.session_state.answers[1] = st.slider("2. I feel energized by social interactions.", 1, 10, st.session_state.answers[1])
    st.session_state.answers[2] = st.slider("3. I prefer working in a team over working alone.", 1, 10, st.session_state.answers[2])
    st.session_state.answers[3] = st.slider("4. I seek out new social experiences.", 1, 10, st.session_state.answers[3])
    st.button("Next", on_click=next_page)

# Page 2 – NS
elif st.session_state.page == 2:
    st.subheader("Page 2: Intuition (N) vs Sensing (S)")
    st.session_state.answers[4] = st.slider("5. I am drawn to abstract concepts.", 1, 10, st.session_state.answers[4])
    st.session_state.answers[5] = st.slider("6. I like to focus on the present moment.", 1, 10, st.session_state.answers[5])
    st.session_state.answers[6] = st.slider("7. I trust my intuition more than facts.", 1, 10, st.session_state.answers[6])
    st.session_state.answers[7] = st.slider("8. I often seek new information to expand my knowledge.", 1, 10, st.session_state.answers[7])
    col1, col2 = st.columns(2)
    col1.button("Back", on_click=prev_page)
    col2.button("Next", on_click=next_page)

# Page 3 – TF
elif st.session_state.page == 3:
    st.subheader("Page 3: Thinking (T) vs Feeling (F)")
    st.session_state.answers[8] = st.slider("9. I enjoy solving logical problems.", 1, 10, st.session_state.answers[8])
    st.session_state.answers[9] = st.slider("10. I value reason over feelings.", 1, 10, st.session_state.answers[9])
    st.session_state.answers[10] = st.slider("11. I prefer to make decisions based on facts.", 1, 10, st.session_state.answers[10])
    col1, col2 = st.columns(2)
    col1.button("Back", on_click=prev_page)
    col2.button("Next", on_click=next_page)

# Page 4 – JP + Submit
elif st.session_state.page == 4:
    st.subheader("Page 4: Judging (J) vs Perceiving (P)")
    st.session_state.answers[11] = st.slider("12. I like structure and organization.", 1, 10, st.session_state.answers[11])
    st.session_state.answers[12] = st.slider("13. I prefer making decisions in a planned and organized way.", 1, 10, st.session_state.answers[12])
    st.session_state.answers[13] = st.slider("14. I like to have a set plan rather than being spontaneous.", 1, 10, st.session_state.answers[13])
    st.session_state.answers[14] = st.slider("15. I feel comfortable following routines.", 1, 10, st.session_state.answers[14])
    col1, col2 = st.columns(2)
    col1.button("Back", on_click=prev_page)
    submitted = col2.button("Submit")

    if submitted:
        answers_array = np.array(st.session_state.answers).reshape(1, -1)
        ie = ie_model.predict(answers_array)[0]
        ns = ns_model.predict(answers_array)[0]
        tf = tf_model.predict(answers_array)[0]
        jp = jp_model.predict(answers_array)[0]
        mbti = get_mbti(ie, ns, tf, jp)

        st.subheader("🎉 Your MBTI Personality Type is: " + mbti)
        st.write(explain_personality_type(mbti))
        st.write(get_advice(mbti))

        # ---- Analytics Section ----
        st.subheader("📊 Quiz Analytics")
        dim_map = [("Introversion", "Extraversion", st.session_state.answers[0:4], ie),
                   ("Intuition", "Sensing", st.session_state.answers[4:8], ns),
                   ("Thinking", "Feeling", st.session_state.answers[8:11], tf),
                   ("Judging", "Perceiving", st.session_state.answers[11:15], jp)]

        # Trait Score Table
        st.markdown("### 📋 Trait Scores")
        trait_table = []
        for trait1, trait2, scores, pred in dim_map:
            avg_score = np.mean(scores)
            trait_table.append([f"{trait1} vs {trait2}", f"{avg_score:.2f}", trait1 if pred == 0 else trait2])
        st.table(pd.DataFrame(trait_table, columns=["Trait", "Avg Score", "Predicted Preference"]))

        # Bar Chart
        labels = [f"{trait1} / {trait2}" for trait1, trait2, _, _ in dim_map]
        avg_scores = [np.mean(scores) for _, _, scores, _ in dim_map]
        plt.figure(figsize=(8, 4))
        sns.barplot(x=labels, y=avg_scores)
        plt.ylim(1, 10)
        plt.ylabel("Average Score")
        plt.title("Average Scores per MBTI Dimension")
        st.pyplot(plt)


        # Radar Chart
        angles = np.linspace(0, 2 * np.pi, len(avg_scores), endpoint=False).tolist()
        avg_scores_radar = avg_scores + avg_scores[:1]
        angles += angles[:1]
        labels_radar = labels + labels[:1]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, avg_scores_radar, 'o-', linewidth=2)
        ax.fill(angles, avg_scores_radar, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels_radar)
        ax.set_title("Personality Radar Chart", y=1.1)
        ax.set_rlim(1, 10)
        st.pyplot(fig)

        # Heatmap
        plt.figure(figsize=(10, 1))
        sns.heatmap([st.session_state.answers], cmap="YlGnBu", annot=True, cbar=False,
                    xticklabels=[f"Q{i+1}" for i in range(15)])
        plt.title("Response Heatmap")
        st.pyplot(plt)

        st.success("✅ Thank you for completing the quiz! Feel free to explore your analytics above.")