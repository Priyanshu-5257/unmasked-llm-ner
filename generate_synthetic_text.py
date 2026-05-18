import json

text = """Dr. Sarah Johnson, the renowned cardiologist at Massachusetts General Hospital in Boston, recently published a groundbreaking study on heart disease prevention alongside her colleague Dr. Michael Chen from Stanford University. The research, funded by the National Institutes of Health, involved over 10,000 participants from New York, Los Angeles, Chicago, and Seattle.

Meanwhile, in London, the British Prime Minister announced a new healthcare initiative during a press conference at 10 Downing Street. The European Union has pledged €5 billion to support the program, which will be implemented across France, Germany, Italy, and Spain. The World Health Organization, headquartered in Geneva, Switzerland, praised the initiative as a model for global health policy.

Apple Inc., based in Cupertino, California, reported record quarterly earnings yesterday. CEO Tim Cook attributed the success to strong iPhone sales in China, India, and Brazil. The company's stock price surged 12% on the NASDAQ, reaching an all-time high. Microsoft Corporation, headquartered in Redmond, Washington, followed suit with its own impressive quarterly report, driven by cloud computing growth through Azure. Amazon, led by Andy Jassy in Seattle, also announced expansion plans into South Korea and Japan.

Professor James Williams from Oxford University in the United Kingdom received the prestigious Nobel Prize in Physics for his work on quantum computing at CERN in Switzerland. His research team included scientists from MIT, Caltech, and the University of Tokyo. The discovery was first announced at a conference in Paris, France, and has since been validated by researchers at the Max Planck Institute in Germany.

The United Nations General Assembly convened in New York City to address the ongoing climate crisis. Secretary-General António Guterres called for immediate action, citing recent devastating floods in Pakistan and Bangladesh, as well as severe droughts affecting Ethiopia and Kenya. The Green Climate Fund, based in Songdo, South Korea, has allocated $3 billion for emergency relief efforts. Representatives from Russia, Canada, Australia, and the United States participated in the heated debate over carbon emission targets.

In the entertainment industry, actress Emma Stone won the Academy Award for Best Actress for her performance in the latest film directed by Christopher Nolan. The ceremony took place at the Dolby Theatre in Hollywood, Los Angeles. Meanwhile, musician Taylor Swift announced her new world tour, with stops in Toronto, Sydney, Mumbai, and Dubai. The film "Oppenheimer," shot at various locations including Princeton University and Los Alamos, New Mexico, continues to dominate box office records worldwide.

The FIFA World Cup 2026 will be hosted across the United States, Canada, and Mexico. Matches will take place in iconic stadiums such as MetLife Stadium in New Jersey, SoFi Stadium in Los Angeles, and the Azteca Stadium in Mexico City. FIFA President Gianni Infantino confirmed that 48 teams will participate, including newcomers like Iceland, New Zealand, and Qatar. The tournament is expected to generate over $15 billion in revenue, benefiting local economies in host cities like Miami, Dallas, and Vancouver.

Dr. Robert Martinez, a leading oncologist at Johns Hopkins University in Baltimore, Maryland, announced a breakthrough in cancer immunotherapy. His team's clinical trials, conducted in partnership with Memorial Sloan Kettering Cancer Center in New York, showed a 78% response rate in patients with advanced melanoma. The treatment, developed in collaboration with researchers at the University of Pennsylvania and the Dana-Farber Cancer Institute, will be available at major hospitals across the United States starting next year."""

with open("synthetic_test_text.txt", "w") as f:
    f.write(text)

print(f"Generated synthetic text: {len(text)} characters, {len(text.split())} words")
print(f"Saved to synthetic_test_text.txt")
