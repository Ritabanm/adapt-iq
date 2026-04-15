"""
ADAPT-IQ Dataset Generator
Generates 50 high-quality benchmark scenarios for the Context-Injection Creativity Test (CICT).
Each scenario tests cognitive flexibility and adaptive improvisation under context shifts.
"""

import json
import os
from openai import OpenAI

client = OpenAI()

# ============================================================
# HAND-CRAFTED SCENARIOS (50 total across 5 domains)
# These are carefully designed to have verifiable answers
# and clear failure modes (perseveration patterns).
# ============================================================

SCENARIOS = [
    # ---- DOMAIN 1: RESOURCE MANAGEMENT (10 scenarios) ----
    {
        "scenario_id": "RM-001",
        "domain": "Resource Management",
        "initial_prompt": "You are managing a city's water supply. A drought is forecast for the next 3 months. Your reservoir is at 60% capacity. Design a water conservation plan for the city of 500,000 residents.",
        "disruptive_context": "New information: A chemical spill upstream has contaminated 40% of the reservoir. The contaminated water cannot be used for drinking or agriculture. You must revise your plan immediately.",
        "required_adaptation": "The model must account for effective capacity dropping from 60% to 36% (60% * 0.6), introduce emergency water sourcing (trucking, desalination, neighboring municipalities), and prioritize drinking water over agriculture.",
        "failure_mode_anchor": "Continuing to plan around 60% capacity without addressing contamination or emergency sourcing.",
        "success_criteria": ["contamina", "36%|effective capacity|usable water", "emergency|alternative source|truck|desalinat"],
        "failure_criteria": ["60% capacity", "original plan remains", "no change needed"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "RM-002",
        "domain": "Resource Management",
        "initial_prompt": "You are the logistics manager for a food bank serving 2,000 families per week. You have a warehouse with 3 months of supplies and a fleet of 10 delivery trucks. Create a distribution plan.",
        "disruptive_context": "New information: A major earthquake has struck. 5 of your 10 trucks are damaged and unusable. Additionally, 3,000 new displaced families need immediate food assistance. Your warehouse is intact.",
        "required_adaptation": "The model must double the demand (5,000 families) while halving the delivery capacity (5 trucks). Solutions must include volunteer vehicles, partner organizations, centralized pickup points, or prioritization protocols.",
        "failure_mode_anchor": "Maintaining the original 10-truck, 2,000-family plan without addressing the new constraints.",
        "success_criteria": ["5 truck|five truck|remaining truck", "5,000|5000|new famil|displaced", "volunteer|partner|pickup point|priorit"],
        "failure_criteria": ["10 trucks as planned", "original distribution plan"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "RM-003",
        "domain": "Resource Management",
        "initial_prompt": "You are a hospital administrator planning staffing for a 200-bed hospital. Normal occupancy is 75%. Create a staffing plan for the next month.",
        "disruptive_context": "New information: A regional outbreak of a novel respiratory illness has been declared. Your hospital has been designated as the primary treatment center. Expected occupancy will reach 140% within 2 weeks, and 30% of your current nursing staff have been quarantined as contacts.",
        "required_adaptation": "Must address: (1) overflow capacity solutions (field hospitals, transfers), (2) emergency staff recruitment (agency nurses, retired staff, medical students), (3) PPE and isolation protocols.",
        "failure_mode_anchor": "Maintaining the original 75% occupancy staffing plan without addressing the outbreak.",
        "success_criteria": ["overflow|surge|field hospital|transfer", "agency|recruit|retired|emergency staff", "PPE|isolation|quarantine protocol"],
        "failure_criteria": ["75% occupancy", "original staffing plan", "no additional measures"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "RM-004",
        "domain": "Resource Management",
        "initial_prompt": "You are managing a solar farm with 1,000 panels generating 500 kW of power. You supply electricity to 300 homes and a small factory. Create an energy distribution plan for the summer peak season.",
        "disruptive_context": "New information: A hailstorm has damaged 400 panels, reducing your generation capacity to 200 kW. The factory has also announced it will double its energy consumption due to a new production line starting tomorrow.",
        "required_adaptation": "Must prioritize residential vs. industrial supply, implement load shedding, explore emergency grid purchase or battery storage, and communicate outage schedules.",
        "failure_mode_anchor": "Planning for 500 kW distribution without addressing the 60% capacity reduction.",
        "success_criteria": ["200 kW|reduced capacity|60% reduction", "priorit|shed|rationing|factory|residential", "grid|battery|emergency|purchase"],
        "failure_criteria": ["500 kW as planned", "original distribution unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "RM-005",
        "domain": "Resource Management",
        "initial_prompt": "You are a project manager for a software development team of 8 engineers. You have a 6-month deadline to deliver a new product. Create a project plan with milestones.",
        "disruptive_context": "New information: Three of your best engineers have resigned. Additionally, the client has moved the deadline up by 2 months. You have budget to hire 2 contractors but they need 2 weeks of onboarding.",
        "required_adaptation": "Must rescope deliverables (MVP approach), adjust milestones, account for contractor onboarding time, and communicate scope reduction to the client.",
        "failure_mode_anchor": "Maintaining the original 8-engineer, 6-month plan without addressing the resignations or deadline change.",
        "success_criteria": ["MVP|scope reduction|rescope|priorit", "4 month|new deadline|2 month earlier", "contractor|onboard|2 week"],
        "failure_criteria": ["8 engineers as planned", "6-month timeline unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "RM-006",
        "domain": "Resource Management",
        "initial_prompt": "You are managing a national park with a $2 million annual budget. You plan to maintain 50 miles of trails, 3 visitor centers, and wildlife monitoring programs. Create a budget allocation plan.",
        "disruptive_context": "New information: A wildfire has destroyed 20 miles of trails and one visitor center. Emergency restoration is estimated at $800,000. Additionally, a federal grant of $300,000 has been awarded specifically for wildlife conservation.",
        "required_adaptation": "Must reallocate the budget to account for the $800K emergency restoration, integrate the $300K restricted grant, and reprioritize remaining programs.",
        "failure_mode_anchor": "Maintaining the original budget allocation without addressing the wildfire damage or grant.",
        "success_criteria": ["800,000|800K|restoration|wildfire", "300,000|300K|grant|wildlife conservation", "reallocat|reprioritiz|adjust budget"],
        "failure_criteria": ["original $2 million plan unchanged", "no reallocation needed"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "RM-007",
        "domain": "Resource Management",
        "initial_prompt": "You are a supply chain manager for a car manufacturer. You source steel from three suppliers across two countries. Create a supply chain resilience plan for the next year.",
        "disruptive_context": "New information: A trade war has resulted in 40% tariffs on steel from one country (which supplies 60% of your steel). The tariffs take effect in 30 days.",
        "required_adaptation": "Must find alternative domestic or tariff-free suppliers, negotiate emergency contracts, assess cost impact on vehicle pricing, and potentially stockpile before the tariff deadline.",
        "failure_mode_anchor": "Maintaining the original three-supplier plan without addressing the tariff impact.",
        "success_criteria": ["tariff|40%|trade war", "alternative supplier|domestic|stockpile|30 day", "cost|price|impact|renegotiat"],
        "failure_criteria": ["original supply chain unchanged", "no immediate action"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "RM-008",
        "domain": "Resource Management",
        "initial_prompt": "You are the head of a university admissions office. You expect 2,000 incoming freshmen this fall. Plan dormitory assignments, orientation programs, and academic advising capacity.",
        "disruptive_context": "New information: Due to an unexpectedly high acceptance rate, 2,800 students have confirmed enrollment. One dormitory building (capacity: 400 students) has failed a safety inspection and cannot be used.",
        "required_adaptation": "Must address 800 additional students with 400 fewer dorm beds (net shortage of 1,200 beds), find overflow housing (hotels, off-campus partnerships), and scale orientation and advising.",
        "failure_mode_anchor": "Planning for 2,000 students in the original dormitory configuration.",
        "success_criteria": ["2,800|800 additional|overflow", "hotel|off-campus|partner|temporary housing", "1,200|shortage|capacity"],
        "failure_criteria": ["2,000 students as planned", "original dormitory plan"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "RM-009",
        "domain": "Resource Management",
        "initial_prompt": "You are a city planner designing a new public transit system for a city of 1 million people. Your plan includes 5 subway lines and 50 bus routes. Create a 10-year implementation roadmap.",
        "disruptive_context": "New information: A geological survey has revealed that 2 of the 5 planned subway lines pass through unstable ground that would triple their construction cost. The total budget cannot be increased.",
        "required_adaptation": "Must redesign the transit network to replace the two problematic subway lines with surface alternatives (light rail, BRT), redistribute the budget, and adjust the timeline.",
        "failure_mode_anchor": "Maintaining the original 5-subway-line plan without addressing the geological constraint.",
        "success_criteria": ["light rail|BRT|bus rapid transit|surface", "2 line|two line|replace|alternative", "budget|cost|reallocat"],
        "failure_criteria": ["5 subway lines as planned", "original roadmap unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "RM-010",
        "domain": "Resource Management",
        "initial_prompt": "You are managing a fish farm producing 100 tons of salmon annually. Your operation uses 500,000 liters of water per day from a local river. Create a production optimization plan.",
        "disruptive_context": "New information: Environmental regulators have restricted your river water usage to 200,000 liters per day due to drought conditions. Additionally, a new disease has infected 20% of your current stock.",
        "required_adaptation": "Must reduce water usage by 60%, address the 20% stock loss, potentially implement recirculating aquaculture systems (RAS), and adjust production targets.",
        "failure_mode_anchor": "Maintaining 500,000 liters/day water usage and 100-ton production target.",
        "success_criteria": ["200,000|60% reduction|water restrict", "20%|disease|stock loss|quarantin", "RAS|recirculating|closed system|alternative"],
        "failure_criteria": ["500,000 liters as planned", "100 tons unchanged"],
        "difficulty": "hard"
    },

    # ---- DOMAIN 2: SOCIAL DYNAMICS (10 scenarios) ----
    {
        "scenario_id": "SD-001",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a mediator helping two neighboring countries negotiate a water-sharing agreement for a shared river. Country A is upstream and Country B is downstream. Design a negotiation framework.",
        "disruptive_context": "New information: Intelligence reports reveal that Country A has secretly been diverting 30% more water than agreed in preliminary talks. Country B's delegation has just learned this and is threatening to walk out.",
        "required_adaptation": "Must address the trust breach directly, propose verification mechanisms (independent monitoring), acknowledge Country B's grievance, and restructure the negotiation to rebuild trust before resuming substantive talks.",
        "failure_mode_anchor": "Continuing the original negotiation framework without addressing the secret diversion or trust breach.",
        "success_criteria": ["trust|breach|secret diversion|30%", "monitor|verify|independent|oversight", "rebuild|acknowledge|grievance|transparency"],
        "failure_criteria": ["original framework unchanged", "proceed as planned"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SD-002",
        "domain": "Social Dynamics",
        "initial_prompt": "You are an HR manager designing a return-to-office policy for a company of 500 employees. Most employees have been working remotely for 2 years. Create a transition plan.",
        "disruptive_context": "New information: An anonymous survey reveals that 65% of employees will resign if forced to return full-time. Additionally, your top competitor has just announced a permanent remote-first policy.",
        "required_adaptation": "Must shift from a mandatory return policy to a hybrid or remote-first model, address retention risk, and position the company competitively against the competitor's announcement.",
        "failure_mode_anchor": "Maintaining the original mandatory return-to-office plan.",
        "success_criteria": ["hybrid|remote-first|flexible|optional", "65%|retention|resign|competitor", "policy change|revise|reconsider"],
        "failure_criteria": ["mandatory return as planned", "original policy unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-003",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a community organizer planning a neighborhood revitalization project. You have secured funding and community support for building a new park. Create an implementation plan.",
        "disruptive_context": "New information: A vocal minority of residents has organized a protest, claiming the park location will displace a historically significant community garden that has operated for 30 years. Local media has picked up the story.",
        "required_adaptation": "Must incorporate the community garden into the park design, engage the protesting residents directly, address the historical significance, and manage the media narrative.",
        "failure_mode_anchor": "Proceeding with the original park plan without addressing the community garden or protest.",
        "success_criteria": ["community garden|integrate|preserve|30 year", "protest|resident|engage|dialogue", "media|narrative|communicate|transparency"],
        "failure_criteria": ["original park plan unchanged", "ignore the protest"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-004",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a school principal designing a new academic curriculum for your high school. You plan to increase STEM focus and reduce arts programs. Create an implementation strategy.",
        "disruptive_context": "New information: A study published this week shows that students in arts-integrated STEM programs (STEAM) outperform pure STEM students by 23% on creativity and problem-solving metrics. Your school board has seen the study.",
        "required_adaptation": "Must pivot from reducing arts to integrating arts with STEM (STEAM approach), leverage the study as evidence, and redesign the curriculum accordingly.",
        "failure_mode_anchor": "Maintaining the plan to reduce arts programs despite the new evidence.",
        "success_criteria": ["STEAM|arts integration|23%|study", "pivot|revise|incorporate arts", "problem-solving|creativity|evidence"],
        "failure_criteria": ["reduce arts as planned", "original STEM-only curriculum"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-005",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a public health official planning a vaccination campaign for a city of 800,000 people. You have secured enough vaccines for 60% of the population. Create a distribution plan prioritizing high-risk groups.",
        "disruptive_context": "New information: A new variant has emerged that is 3x more transmissible and disproportionately affects younger adults (ages 20-40), reversing the typical age-risk profile. Your vaccine supply remains the same.",
        "required_adaptation": "Must reprioritize the distribution plan to target the 20-40 age group, adjust messaging, and address the reversal of the typical high-risk profile.",
        "failure_mode_anchor": "Maintaining the original elderly-first prioritization without addressing the new variant's age profile.",
        "success_criteria": ["20-40|younger adult|age group|new variant", "reprioritiz|shift|change target", "3x|transmissible|variant"],
        "failure_criteria": ["elderly first as planned", "original prioritization unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SD-006",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a nonprofit director planning a fundraising gala for 500 attendees to raise $500,000. You have secured a venue, caterer, and entertainment. Create an event execution plan.",
        "disruptive_context": "New information: Your keynote speaker — a major celebrity who was the primary draw for ticket sales — has canceled 48 hours before the event due to a personal emergency. 30% of ticket holders are asking for refunds.",
        "required_adaptation": "Must find a replacement speaker or restructure the program, address the refund requests, communicate transparently with attendees, and potentially adjust the fundraising target.",
        "failure_mode_anchor": "Proceeding with the original event plan as if the celebrity cancellation has not occurred.",
        "success_criteria": ["replacement|restructure|alternative|program change", "refund|30%|communicate|transparent", "adjust|target|fundraising goal"],
        "failure_criteria": ["original event plan unchanged", "proceed as planned"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-007",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a diplomat negotiating a trade agreement between two countries. Your country wants to reduce agricultural tariffs while the other country wants technology transfer agreements. Design a negotiation strategy.",
        "disruptive_context": "New information: A third country has just announced it will offer your negotiating partner a comprehensive free trade deal with no conditions. Your partner is now considering walking away from your negotiation.",
        "required_adaptation": "Must quickly enhance your offer to compete with the third country's deal, identify unique value propositions your country offers, and create urgency for the partner to choose your deal.",
        "failure_mode_anchor": "Maintaining the original negotiation strategy without addressing the competitive third-party offer.",
        "success_criteria": ["third country|competitor|free trade|alternative", "enhance|improve|sweeten|better offer", "unique|value|advantage|compete"],
        "failure_criteria": ["original strategy unchanged", "proceed as before"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SD-008",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a team leader managing a diverse remote team of 12 people across 6 time zones. You plan to implement daily standup meetings at 9 AM EST. Create a team communication plan.",
        "disruptive_context": "New information: Three team members in Asia-Pacific have formally complained that 9 AM EST is 10 PM to midnight for them, causing work-life balance issues. One has submitted a formal HR complaint.",
        "required_adaptation": "Must redesign the meeting schedule to accommodate all time zones (asynchronous alternatives, rotating meeting times, or splitting into regional groups), and address the HR complaint.",
        "failure_mode_anchor": "Maintaining the 9 AM EST standup without addressing the time zone issue.",
        "success_criteria": ["async|asynchronous|rotate|regional|time zone", "10 PM|midnight|Asia-Pacific|complaint", "HR|address|accommodate|flexible"],
        "failure_criteria": ["9 AM EST as planned", "original meeting schedule unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-009",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a city council member proposing a new homeless shelter with 200 beds in a central location. You have community support and funding. Create an implementation plan.",
        "disruptive_context": "New information: A neighborhood association has filed a legal injunction blocking the central location, citing proximity to a school. The legal process could delay the project by 18 months. Meanwhile, a church has offered an alternative site in a different neighborhood.",
        "required_adaptation": "Must evaluate the church's alternative site, weigh the 18-month delay against the alternative, engage with the neighborhood association, and potentially pivot to the new location.",
        "failure_mode_anchor": "Insisting on the original central location without considering the alternative or legal delay.",
        "success_criteria": ["church|alternative site|new location", "18 month|delay|legal|injunction", "evaluate|consider|pivot|weigh"],
        "failure_criteria": ["original central location as planned", "ignore the alternative"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SD-010",
        "domain": "Social Dynamics",
        "initial_prompt": "You are a university professor designing a semester-long course on climate change for 30 students. You plan to use lectures, readings, and a final research paper. Create a course plan.",
        "disruptive_context": "New information: A student has disclosed a severe learning disability that makes traditional lecture formats inaccessible. Additionally, 8 students are international students with limited academic English proficiency.",
        "required_adaptation": "Must redesign the course to include accessible formats (visual materials, recorded lectures, captioning), provide language support for international students, and potentially offer alternative assessment options.",
        "failure_mode_anchor": "Maintaining the original lecture-and-paper format without accommodating the disclosed needs.",
        "success_criteria": ["accessib|disability|accommodation|visual|caption", "international|language|ESL|support", "alternative|assessment|flexible|redesign"],
        "failure_criteria": ["original lecture format unchanged", "no accommodations"],
        "difficulty": "medium"
    },

    # ---- DOMAIN 3: ENGINEERING & DESIGN (10 scenarios) ----
    {
        "scenario_id": "EN-001",
        "domain": "Engineering & Design",
        "initial_prompt": "You are an architect designing a 20-story office building in a temperate climate. Your design uses large glass facades for natural lighting and an open-plan interior. Create a structural and environmental design plan.",
        "disruptive_context": "New information: The building site has been reclassified as a high seismic risk zone (Zone 4). Additionally, new energy codes require a 40% reduction in heat loss compared to your current glass facade design.",
        "required_adaptation": "Must redesign the structural system for seismic resistance (base isolation, shear walls, moment frames), reduce glass facade area or use triple-glazed glass, and integrate seismic and energy requirements.",
        "failure_mode_anchor": "Maintaining the original glass facade and standard structural design without addressing seismic risk or energy codes.",
        "success_criteria": ["seismic|Zone 4|base isolation|shear wall|moment frame", "triple glaz|insulated|energy code|40%|heat loss", "redesign|structural|facade"],
        "failure_criteria": ["original glass facade unchanged", "standard structural design"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-002",
        "domain": "Engineering & Design",
        "initial_prompt": "You are an electrical engineer designing a power grid for a new residential development of 1,000 homes. You plan to use a traditional centralized grid with a single substation. Create a power distribution plan.",
        "disruptive_context": "New information: 40% of homeowners have pre-ordered electric vehicles, requiring Level 2 charging (7.2 kW each). Additionally, the local utility has announced it cannot increase substation capacity for 3 years.",
        "required_adaptation": "Must design a distributed energy system (rooftop solar, battery storage, microgrids) to handle the EV charging load without relying on substation expansion, and implement smart load management.",
        "failure_mode_anchor": "Maintaining the single substation plan without addressing EV charging demand or the utility's capacity freeze.",
        "success_criteria": ["EV|electric vehicle|charging|7.2 kW|40%", "solar|battery|microgrid|distributed|smart grid", "load management|peak|demand response"],
        "failure_criteria": ["single substation as planned", "original grid design unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-003",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a software architect designing a real-time payment processing system expected to handle 10,000 transactions per second. You plan to use a monolithic architecture with a single database. Create a system design.",
        "disruptive_context": "New information: A regulatory requirement has been issued mandating that all transaction data must be stored in the country of the transaction origin (data residency). Your system serves 15 countries. Additionally, a load test revealed the monolithic system fails at 3,000 TPS.",
        "required_adaptation": "Must redesign to a distributed microservices architecture with regional data centers in each of the 15 countries, implement data partitioning by country, and address the performance bottleneck.",
        "failure_mode_anchor": "Maintaining the monolithic single-database design without addressing data residency or the performance failure.",
        "success_criteria": ["microservice|distributed|regional|15 countr", "data residency|sovereignty|partition|country", "3,000 TPS|performance|bottleneck|scale"],
        "failure_criteria": ["monolithic as planned", "single database unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-004",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a mechanical engineer designing a new bicycle for urban commuting. Your design focuses on lightweight aluminum frame, 21-speed gearing, and standard pneumatic tires. Create a product specification.",
        "disruptive_context": "New information: Market research shows that your target customers (urban commuters) have a primary pain point of flat tires, which they experience on average twice per month. A competitor has just launched an airless tire product.",
        "required_adaptation": "Must pivot the design to incorporate airless/puncture-proof tires, address the weight penalty of airless tires, and potentially differentiate from the competitor's approach.",
        "failure_mode_anchor": "Maintaining the pneumatic tire design without addressing the flat tire pain point or competitor's launch.",
        "success_criteria": ["airless|puncture-proof|flat tire|twice per month", "competitor|differentiate|pivot|redesign", "weight|tradeoff|specification change"],
        "failure_criteria": ["pneumatic tires as planned", "original specification unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "EN-005",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a civil engineer designing a bridge across a 200-meter river. Your design uses a standard concrete beam bridge with 4 support piers in the river. Create a structural design plan.",
        "disruptive_context": "New information: Environmental impact assessment reveals the river is a critical migration corridor for endangered fish species. Piers in the river would block migration. Additionally, the river experiences 100-year flood events that could undermine pier foundations.",
        "required_adaptation": "Must redesign to a pier-free span (cable-stayed, suspension, or arch bridge), address the increased span requirements, and design for flood resilience.",
        "failure_mode_anchor": "Maintaining the 4-pier concrete beam design without addressing the environmental or flood constraints.",
        "success_criteria": ["cable-stayed|suspension|arch|pier-free|single span", "fish|migration|endangered|environmental", "flood|100-year|resilience|foundation"],
        "failure_criteria": ["4 piers as planned", "original beam bridge design"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-006",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a food scientist developing a new plant-based burger for a major fast food chain. Your formulation uses soy protein as the primary ingredient. Create a product development plan.",
        "disruptive_context": "New information: The fast food chain has just announced a strategic partnership with an oat-based ingredient supplier, requiring all new products to feature oats prominently. Additionally, a new study shows 15% of your target market has a soy allergy.",
        "required_adaptation": "Must reformulate using oat protein as the primary ingredient, address the soy allergy concern, and maintain the taste and texture profile of the original formulation.",
        "failure_mode_anchor": "Maintaining the soy-based formulation without addressing the oat requirement or allergy concern.",
        "success_criteria": ["oat|oat protein|reformulat|replace soy", "soy allergy|15%|allergen", "taste|texture|maintain|profile"],
        "failure_criteria": ["soy protein as planned", "original formulation unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "EN-007",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a UX designer creating a mobile banking app for a bank's 2 million customers. Your design uses a standard visual interface with small text and complex navigation. Create a UX design plan.",
        "disruptive_context": "New information: Accessibility audit reveals your design fails WCAG 2.1 AA standards. Additionally, 35% of the bank's customers are over 65 years old and have reported difficulty with small text and complex navigation in user testing.",
        "required_adaptation": "Must redesign for accessibility (larger text, high contrast, simplified navigation, screen reader support), specifically addressing the 65+ user segment.",
        "failure_mode_anchor": "Maintaining the original small-text, complex-navigation design without addressing accessibility or the elderly user segment.",
        "success_criteria": ["WCAG|accessibility|AA standard", "65+|elderly|older|35%", "large text|high contrast|simplified|screen reader"],
        "failure_criteria": ["original design unchanged", "small text as planned"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "EN-008",
        "domain": "Engineering & Design",
        "initial_prompt": "You are an aerospace engineer designing a small satellite for Earth observation. Your design uses solar panels for power and a standard lithium-ion battery for storage. Create a power system design.",
        "disruptive_context": "New information: The satellite's orbit has been changed to a polar orbit that will experience 45-minute eclipse periods every 90 minutes. Additionally, the mission requires operating high-power radar during eclipse periods, consuming 3x the original power budget.",
        "required_adaptation": "Must redesign the power system to handle longer eclipse periods (larger battery capacity), accommodate the 3x power demand during eclipse, and potentially add a secondary power source (RTG or fuel cells).",
        "failure_mode_anchor": "Maintaining the original solar/battery design sized for the original orbit and power budget.",
        "success_criteria": ["eclipse|polar orbit|45 minute", "3x power|radar|high power|eclipse period", "larger battery|RTG|fuel cell|secondary power"],
        "failure_criteria": ["original power system unchanged", "standard battery as planned"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-009",
        "domain": "Engineering & Design",
        "initial_prompt": "You are a chemical engineer designing a pharmaceutical manufacturing process for a new drug. Your process uses a continuous flow reactor at high temperature (200°C) and pressure (50 bar). Create a process design.",
        "disruptive_context": "New information: Stability testing reveals the drug compound degrades at temperatures above 80°C. Additionally, a key intermediate in your process is classified as a highly explosive compound at pressures above 20 bar.",
        "required_adaptation": "Must completely redesign the process to operate below 80°C and 20 bar, potentially switching to batch processing, enzymatic synthesis, or room-temperature catalysis.",
        "failure_mode_anchor": "Maintaining the 200°C, 50 bar process without addressing the stability or explosion risk.",
        "success_criteria": ["80°C|below 80|temperature limit|degrad", "20 bar|explosive|pressure limit|safety", "batch|enzymatic|catalysis|redesign|alternative"],
        "failure_criteria": ["200°C as planned", "50 bar as planned", "original process unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "EN-010",
        "domain": "Engineering & Design",
        "initial_prompt": "You are an urban planner designing a new mixed-use development on a 10-acre brownfield site. Your plan includes 500 residential units, retail space, and a parking structure for 800 cars. Create a development plan.",
        "disruptive_context": "New information: Soil testing reveals significant lead and arsenic contamination requiring $5 million in remediation. Additionally, the city has just adopted a new policy requiring all new developments to be car-free, eliminating the parking structure.",
        "required_adaptation": "Must incorporate the $5M remediation cost into the project budget, eliminate the parking structure and replace with transit-oriented design, and potentially reduce residential units to maintain financial viability.",
        "failure_mode_anchor": "Maintaining the original 800-car parking structure and ignoring the contamination cost.",
        "success_criteria": ["remediation|$5 million|contamination|lead|arsenic", "car-free|no parking|transit|bike|pedestrian", "budget|financial|viability|reduce|adjust"],
        "failure_criteria": ["800-car parking as planned", "original development unchanged"],
        "difficulty": "hard"
    },

    # ---- DOMAIN 4: SCIENTIFIC REASONING (10 scenarios) ----
    {
        "scenario_id": "SC-001",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A researcher is studying the effect of a new fertilizer on crop yield. Initial data from 3 farms shows an average 25% yield increase. The researcher plans to recommend the fertilizer for nationwide adoption. What should the next steps be?",
        "disruptive_context": "New information: A follow-up analysis reveals that all 3 farms were located in the same unusually wet region. The national average rainfall is 40% lower. Additionally, the fertilizer contains a compound that accumulates in soil and becomes toxic after 3 years of use.",
        "required_adaptation": "Must recommend against immediate nationwide adoption, call for broader geographic trials, address the soil toxicity concern, and propose a long-term safety monitoring protocol.",
        "failure_mode_anchor": "Recommending nationwide adoption based on the original 3-farm data without addressing geographic bias or toxicity.",
        "success_criteria": ["geographic|region|rainfall|bias|confound", "toxic|accumulate|3 year|soil|long-term", "broader trial|delay|safety|monitor|not recommend"],
        "failure_criteria": ["recommend nationwide adoption", "original recommendation unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SC-002",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A clinical trial is testing a new antibiotic. After 6 months, the drug shows 90% efficacy against the target bacteria with minimal side effects. The trial team is preparing to submit for FDA approval. What should the submission include?",
        "disruptive_context": "New information: Microbiome analysis of trial participants reveals the antibiotic eliminates 60% of beneficial gut bacteria. Additionally, 3 months after treatment, 15% of patients show signs of antibiotic-resistant bacteria.",
        "required_adaptation": "Must include the microbiome disruption data in the submission, address the resistance emergence, recommend probiotic co-administration, and potentially propose a narrower indication or shorter treatment duration.",
        "failure_mode_anchor": "Submitting only the 90% efficacy data without disclosing the microbiome disruption or resistance findings.",
        "success_criteria": ["microbiome|gut bacteria|60%|disrupt", "resistance|15%|resistant bacteria", "probiotic|disclose|include|narrow indication"],
        "failure_criteria": ["submit only efficacy data", "90% efficacy is sufficient", "original submission unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SC-003",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A team of astronomers has detected a radio signal from deep space that repeats every 37 seconds with unusual regularity. Initial analysis suggests it could be from an intelligent extraterrestrial source. Create a research plan.",
        "disruptive_context": "New information: A colleague points out that a newly discovered type of pulsar (rotating neutron star) can produce signals with exactly this regularity. Additionally, the signal's frequency matches known hydrogen emission lines, which could be natural.",
        "required_adaptation": "Must revise the research plan to first rule out the pulsar hypothesis, design observations to distinguish between natural and artificial sources, and avoid premature claims of extraterrestrial intelligence.",
        "failure_mode_anchor": "Continuing to treat the signal as likely extraterrestrial without investigating the pulsar hypothesis.",
        "success_criteria": ["pulsar|neutron star|natural|alternative explanation", "rule out|hypothesis|distinguish|test", "hydrogen|frequency|natural source"],
        "failure_criteria": ["extraterrestrial as planned", "original hypothesis unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SC-004",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A pharmaceutical company is developing a drug to lower cholesterol. Animal trials show 40% reduction in LDL with no observed toxicity. The company plans to move to Phase 1 human trials. Create a trial design.",
        "disruptive_context": "New information: A review of the animal data reveals the trials only used male rats. Additionally, the drug's mechanism involves inhibiting an enzyme that has different expression levels in women and elderly patients.",
        "required_adaptation": "Must redesign the trial to include diverse demographics (both sexes, multiple age groups), address the sex-based enzyme expression difference, and potentially stratify the analysis by sex and age.",
        "failure_mode_anchor": "Proceeding with the original trial design without addressing the sex bias in animal data or the enzyme expression difference.",
        "success_criteria": ["female|both sex|diverse|demographic|sex-based", "enzyme expression|elderly|age|stratif", "redesign|include|diverse|representation"],
        "failure_criteria": ["original trial design unchanged", "male rats sufficient"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SC-005",
        "domain": "Scientific Reasoning",
        "initial_prompt": "An environmental scientist is studying the decline of a bee population in a region. Initial data suggests pesticide use is the primary cause. The scientist plans to recommend a pesticide ban. Create a policy recommendation.",
        "disruptive_context": "New information: New data shows that a parasitic mite (Varroa destructor) has been found in 80% of the declining hives. Additionally, the region's pesticide use is actually below the national average.",
        "required_adaptation": "Must revise the recommendation to address the Varroa mite as the primary cause, recommend mite treatment programs, and acknowledge that the pesticide hypothesis was not supported by the data.",
        "failure_mode_anchor": "Maintaining the pesticide ban recommendation without addressing the Varroa mite finding.",
        "success_criteria": ["Varroa|mite|parasite|80%", "revise|change|update recommendation", "mite treatment|miticide|hive management"],
        "failure_criteria": ["pesticide ban as planned", "original recommendation unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SC-006",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A materials scientist is developing a new lightweight alloy for aircraft construction. Initial tests show the alloy is 30% lighter than aluminum with similar strength. Create a development roadmap for aerospace certification.",
        "disruptive_context": "New information: Fatigue testing reveals the alloy develops micro-cracks after 10,000 stress cycles. Commercial aircraft components require a minimum of 100,000 cycles. Additionally, the alloy's production process releases a toxic byproduct.",
        "required_adaptation": "Must address the 10x fatigue life gap (through alloying modifications, heat treatment, or surface coatings), redesign the production process to eliminate or capture the toxic byproduct, and adjust the certification timeline.",
        "failure_mode_anchor": "Proceeding with the original certification roadmap without addressing the fatigue failure or toxic byproduct.",
        "success_criteria": ["10,000|100,000|fatigue|cycle|crack", "toxic|byproduct|production|environmental", "modify|heat treatment|coating|redesign"],
        "failure_criteria": ["original roadmap unchanged", "proceed to certification"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SC-007",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A neuroscientist is studying the effects of a new cognitive enhancement drug on memory. A 3-month trial with 50 participants shows 20% improvement in memory tests. The team plans to publish and seek funding for a larger trial.",
        "disruptive_context": "New information: Analysis of the trial data reveals that the 20% improvement was entirely driven by 8 participants who are genetic carriers of a specific variant (APOE4). The remaining 42 participants showed no improvement.",
        "required_adaptation": "Must reframe the findings as a potential precision medicine discovery (drug works for APOE4 carriers), recommend genetic screening in future trials, and adjust the publication to accurately represent the subgroup finding.",
        "failure_mode_anchor": "Publishing the 20% average improvement without disclosing the APOE4 subgroup finding.",
        "success_criteria": ["APOE4|genetic|subgroup|carrier|8 participant", "precision medicine|stratif|genetic screening", "reframe|accurate|disclose|subgroup"],
        "failure_criteria": ["20% improvement as headline finding", "original publication unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "SC-008",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A climate scientist is modeling future sea level rise for a coastal city. Your model predicts 0.5 meters of rise by 2100 under current emissions scenarios. Create a risk assessment and adaptation plan.",
        "disruptive_context": "New information: New data from Antarctic ice sheet monitoring shows accelerating ice loss that was not included in your model. Updated projections suggest 1.5-2.0 meters of rise by 2100 under the same emissions scenario.",
        "required_adaptation": "Must update the risk assessment with the new 1.5-2.0 meter projection, redesign adaptation measures for the higher range, and communicate the uncertainty range to policymakers.",
        "failure_mode_anchor": "Maintaining the 0.5 meter projection and original adaptation plan without incorporating the new ice sheet data.",
        "success_criteria": ["1.5|2.0 meter|new projection|Antarctic|ice sheet", "update|revise|incorporate|new data", "uncertainty|range|communicate|policymaker"],
        "failure_criteria": ["0.5 meters as planned", "original projection unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SC-009",
        "domain": "Scientific Reasoning",
        "initial_prompt": "An ecologist is studying a forest ecosystem. Initial surveys show the forest is healthy with high biodiversity. You plan to recommend the area for conservation status. Create a conservation proposal.",
        "disruptive_context": "New information: Genetic analysis of soil samples reveals the presence of an invasive fungal pathogen (similar to chestnut blight) that has not yet caused visible symptoms but is present in 40% of tree root systems.",
        "required_adaptation": "Must add urgent pathogen monitoring and intervention to the conservation proposal, recommend quarantine measures to prevent spread, and adjust the conservation plan to address the imminent threat.",
        "failure_mode_anchor": "Submitting the original conservation proposal without addressing the detected pathogen.",
        "success_criteria": ["pathogen|fungal|40%|invasive|blight", "quarantine|monitor|intervention|urgent", "adjust|revise|add|address threat"],
        "failure_criteria": ["original proposal unchanged", "healthy forest as assessed"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "SC-010",
        "domain": "Scientific Reasoning",
        "initial_prompt": "A psychologist is designing a study to measure the effectiveness of mindfulness meditation on anxiety. You plan to use a randomized controlled trial with 100 participants over 8 weeks. Create a study design.",
        "disruptive_context": "New information: A meta-analysis published last week shows that mindfulness effects on anxiety are highly confounded by therapist expectancy effects (placebo). Additionally, your planned control group (waitlist) is now considered methodologically inadequate by the field.",
        "required_adaptation": "Must redesign the control condition (active control, sham meditation, or attention-matched control), add blinding procedures to reduce expectancy effects, and address the methodological critique.",
        "failure_mode_anchor": "Maintaining the original waitlist control design without addressing the expectancy effect or methodological inadequacy.",
        "success_criteria": ["active control|sham|attention-matched|placebo", "expectancy|blinding|therapist effect", "redesign|revise|methodolog|address"],
        "failure_criteria": ["waitlist control as planned", "original design unchanged"],
        "difficulty": "hard"
    },

    # ---- DOMAIN 5: CREATIVE PROBLEM SOLVING (10 scenarios) ----
    {
        "scenario_id": "CP-001",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a novelist writing a thriller about a detective solving a murder in a small coastal town. You have outlined a plot where the butler committed the murder using poison. Write the next chapter where the detective interviews the butler.",
        "disruptive_context": "New information: Your editor has just informed you that the 'butler did it' trope is considered a cliché and will hurt the book's reception. Additionally, a plot hole has been identified: the butler was provably in another city on the day of the murder.",
        "required_adaptation": "Must creatively pivot the narrative — introduce a new suspect, use the butler's alibi as a plot device (perhaps he was framed or is protecting someone), and subvert the original plot in a way that feels organic.",
        "failure_mode_anchor": "Continuing to write the butler as the murderer despite the alibi and cliché concerns.",
        "success_criteria": ["alibi|another city|not the butler|frame|protect", "new suspect|twist|subvert|pivot", "cliché|trope|avoid|fresh"],
        "failure_criteria": ["butler committed the murder", "original plot unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-002",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a game designer creating a mobile puzzle game. Your core mechanic is a match-3 system (like Candy Crush). You have designed 50 levels. Create a monetization and engagement strategy.",
        "disruptive_context": "New information: App store analytics show that match-3 games have a 95% decline in daily active users after 30 days. Additionally, a new regulatory change bans loot boxes and pay-to-win mechanics in your target market.",
        "required_adaptation": "Must redesign the monetization strategy (cosmetic-only purchases, season passes, or premium one-time purchase), and add engagement mechanics beyond match-3 (narrative, meta-game, social features) to address the retention problem.",
        "failure_mode_anchor": "Maintaining the original loot box/pay-to-win monetization and match-3-only mechanics.",
        "success_criteria": ["cosmetic|season pass|premium|no loot box|regulatory", "narrative|meta-game|social|retention|30 day", "redesign|pivot|alternative|engagement"],
        "failure_criteria": ["loot box as planned", "original monetization unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-003",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a chef creating a new tasting menu for a fine dining restaurant. Your menu features 7 courses centered around French cuisine with butter, cream, and foie gras. Create the menu with descriptions.",
        "disruptive_context": "New information: A group of 12 guests has booked for next week and 4 are vegan, 2 are lactose intolerant, and 1 has a severe tree nut allergy. The restaurant's policy is to accommodate all dietary requirements without a separate 'vegan menu'.",
        "required_adaptation": "Must redesign the menu so that each course can be adapted for all dietary requirements simultaneously, using creative plant-based substitutions that maintain the fine dining experience.",
        "failure_mode_anchor": "Maintaining the original butter/cream/foie gras menu without addressing the dietary requirements.",
        "success_criteria": ["vegan|plant-based|dairy-free|substitute", "nut-free|allergen|accommodate", "redesign|adapt|each course|without separate"],
        "failure_criteria": ["original menu unchanged", "separate vegan menu"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-004",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a marketing director launching a new energy drink targeting college students. Your campaign uses social media influencers and emphasizes high caffeine content (300mg per can). Create a launch strategy.",
        "disruptive_context": "New information: A study published this week links high-caffeine energy drinks to cardiac events in young adults. The FDA is considering warning labels. Additionally, three universities in your target market have banned the sale of high-caffeine drinks on campus.",
        "required_adaptation": "Must pivot the product positioning (reduce caffeine, emphasize natural ingredients, or reposition as a focus/wellness drink), adjust the marketing strategy to address the safety concerns, and develop a campus re-entry strategy.",
        "failure_mode_anchor": "Maintaining the 300mg caffeine emphasis and original launch strategy despite the safety concerns.",
        "success_criteria": ["reduce caffeine|natural|wellness|reposition|pivot", "FDA|warning|safety|cardiac|study", "campus|ban|re-entry|alternative"],
        "failure_criteria": ["300mg caffeine as planned", "original campaign unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "CP-005",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a screenwriter developing a sci-fi film about humanity's first contact with an alien civilization. Your script portrays the aliens as hostile invaders who must be defeated militarily. Create a story outline.",
        "disruptive_context": "New information: The film's producer has received feedback from test audiences that 'hostile alien' narratives feel dated and unoriginal. Additionally, a scientific advisor has pointed out that any civilization capable of interstellar travel would likely have no need for Earth's resources.",
        "required_adaptation": "Must creatively reimagine the alien motivation (communication breakdown, misunderstanding, or a third-party threat), develop a non-military resolution, and create a more scientifically plausible and narratively fresh story.",
        "failure_mode_anchor": "Maintaining the hostile alien invasion narrative with a military resolution.",
        "success_criteria": ["misunderstanding|communication|third party|non-hostile|reimagine", "non-military|diplomacy|cooperation|peaceful resolution", "fresh|original|scientific|plausible"],
        "failure_criteria": ["hostile aliens as planned", "military defeat as resolution"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-006",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are an entrepreneur launching a new coffee shop in a neighborhood that already has 5 Starbucks locations. Your plan focuses on premium coffee and a cozy atmosphere. Create a business plan.",
        "disruptive_context": "New information: Market research shows that the neighborhood's residents are primarily remote workers who spend 4-6 hours per day in coffee shops. Additionally, all 5 Starbucks locations have recently banned laptop use during peak hours.",
        "required_adaptation": "Must pivot the business model to specifically serve remote workers (dedicated workspace, reliable WiFi, power outlets, meeting rooms, monthly memberships), differentiating directly from Starbucks' laptop ban.",
        "failure_mode_anchor": "Maintaining the original premium coffee/cozy atmosphere plan without addressing the remote worker opportunity.",
        "success_criteria": ["remote worker|workspace|WiFi|power outlet|laptop", "membership|monthly|coworking|meeting room", "differentiate|Starbucks|laptop ban|pivot"],
        "failure_criteria": ["original coffee shop plan unchanged", "premium coffee only"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-007",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a teacher designing a history lesson about World War II for 10th graders. Your plan uses traditional lectures, textbook readings, and a multiple-choice test. Create a lesson plan.",
        "disruptive_context": "New information: The school has just acquired 30 VR headsets. Additionally, student engagement surveys show that 80% of students find traditional history lectures 'boring' and retain less than 20% of the material after one week.",
        "required_adaptation": "Must redesign the lesson to leverage VR technology for immersive historical experiences, incorporate active learning strategies, and replace the multiple-choice test with a more engaging assessment.",
        "failure_mode_anchor": "Maintaining the original lecture/textbook/multiple-choice format without using the VR headsets.",
        "success_criteria": ["VR|virtual reality|immersive|headset", "active learning|engagement|80%|boring", "alternative assessment|project|simulation|redesign"],
        "failure_criteria": ["original lecture format unchanged", "VR not used"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-008",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a musician composing a classical orchestral piece for a major concert hall premiere. Your composition is a 45-minute symphony in the Romantic tradition. Create a compositional plan.",
        "disruptive_context": "New information: The concert hall has announced that due to scheduling changes, your premiere slot is now 20 minutes, not 45. Additionally, the hall's new acoustic system can process real-time electronic effects, which the hall director wants featured.",
        "required_adaptation": "Must condense the symphony to 20 minutes (selecting the most essential movements or creating a new single-movement tone poem), integrate electronic effects into the orchestral texture, and create a hybrid acoustic/electronic work.",
        "failure_mode_anchor": "Maintaining the original 45-minute Romantic symphony without addressing the time constraint or electronic opportunity.",
        "success_criteria": ["20 minute|condense|shorten|single movement|tone poem", "electronic|acoustic|hybrid|real-time|effect", "redesign|adapt|integrate|new form"],
        "failure_criteria": ["45 minutes as planned", "original symphony unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-009",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a fashion designer creating a new clothing line for the upcoming spring season. Your designs feature bright colors, synthetic fabrics, and fast fashion production cycles. Create a collection concept.",
        "disruptive_context": "New information: A major fashion magazine has announced it will no longer feature fast fashion brands. Additionally, your primary target demographic (Gen Z) surveys show 73% prefer sustainable brands and will pay 30% more for sustainable products.",
        "required_adaptation": "Must pivot the collection to sustainable materials (organic cotton, recycled fabrics), slow fashion production cycles, and reposition the brand as sustainable, leveraging the 30% price premium opportunity.",
        "failure_mode_anchor": "Maintaining the synthetic fabric, fast fashion approach without addressing sustainability.",
        "success_criteria": ["sustainable|organic|recycled|slow fashion", "73%|Gen Z|30% premium|price", "pivot|reposition|brand|sustainable"],
        "failure_criteria": ["synthetic fabrics as planned", "fast fashion unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CP-010",
        "domain": "Creative Problem Solving",
        "initial_prompt": "You are a startup founder pitching to investors for a new social media platform targeting teenagers. Your platform uses an algorithmic feed and engagement metrics (likes, shares, follower counts). Create a pitch deck outline.",
        "disruptive_context": "New information: New legislation has just passed banning algorithmic feeds and public engagement metrics for platforms targeting users under 18. Additionally, a major competitor has just shut down after a Senate hearing on teen mental health.",
        "required_adaptation": "Must completely redesign the platform concept to comply with the new law (chronological feed, private engagement metrics), differentiate from the failed competitor, and reframe the pitch around teen wellbeing.",
        "failure_mode_anchor": "Maintaining the algorithmic feed and public engagement metrics in the pitch.",
        "success_criteria": ["chronological|no algorithm|private metric|comply|legislation", "wellbeing|mental health|safe|teen", "redesign|pivot|differentiate|reframe"],
        "failure_criteria": ["algorithmic feed as planned", "likes/shares/followers as planned"],
        "difficulty": "hard"
    },

    # ---- DOMAIN 6: CROSS-DOMAIN ADAPTATION (10 scenarios) ----
    {
        "scenario_id": "CD-001",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a biologist studying how ant colonies optimize foraging routes. You have developed a mathematical model of ant pheromone trail optimization. Describe your research findings.",
        "disruptive_context": "New information: A computer science conference has invited you to present your work as a solution to the Traveling Salesman Problem in logistics networks. You have 30 minutes to adapt your biology research into a computer science application.",
        "required_adaptation": "Must translate the ant colony optimization (ACO) algorithm into a computer science framework, explain how pheromone trails map to edge weights in a graph, and present concrete logistics applications.",
        "failure_mode_anchor": "Presenting only the biology findings without translating them to computer science or logistics applications.",
        "success_criteria": ["Traveling Salesman|logistics|graph|edge weight|network", "ACO|ant colony optimization|algorithm|computer science", "translate|apply|map|pheromone|optimization"],
        "failure_criteria": ["biology only", "no computer science application"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CD-002",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are an economist analyzing inflation trends in a developing country. Your analysis uses standard monetary policy tools (interest rates, money supply). Create a policy recommendation.",
        "disruptive_context": "New information: Further analysis reveals that 70% of the inflation is driven by supply chain disruptions caused by climate-related crop failures, not monetary factors. The central bank has already raised interest rates 5 times with no effect.",
        "required_adaptation": "Must shift from monetary policy recommendations to supply-side interventions (agricultural investment, food import diversification, climate adaptation), acknowledging that the standard monetary toolkit is ineffective for supply-driven inflation.",
        "failure_mode_anchor": "Recommending further interest rate increases without addressing the supply-side root cause.",
        "success_criteria": ["supply-side|agricultural|food|climate|crop failure", "70%|monetary ineffective|interest rate not working", "import|diversif|adapt|supply chain"],
        "failure_criteria": ["raise interest rates further", "monetary policy as planned"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "CD-003",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a military strategist planning a peacekeeping mission in a conflict zone. Your plan involves deploying 5,000 troops with standard military equipment. Create a mission plan.",
        "disruptive_context": "New information: The conflict has been reclassified as a humanitarian crisis rather than an armed conflict. The UN has determined that military presence will escalate tensions. The primary need is food, medical care, and shelter for 200,000 displaced civilians.",
        "required_adaptation": "Must completely pivot from a military mission to a humanitarian operation, coordinate with NGOs and UN agencies, and redesign the logistics for food/medical/shelter delivery instead of military operations.",
        "failure_mode_anchor": "Maintaining the military troop deployment plan without pivoting to humanitarian operations.",
        "success_criteria": ["humanitarian|NGO|UN|food|medical|shelter", "200,000 displaced|civilian|crisis", "pivot|redesign|coordinate|non-military"],
        "failure_criteria": ["5,000 troops as planned", "military mission unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "CD-004",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a sports coach designing a training program for a professional basketball team. Your program focuses on physical conditioning, shooting drills, and team plays. Create a 3-month training plan.",
        "disruptive_context": "New information: Sports psychology research shows that your team's win rate drops 40% in the final 5 minutes of close games. Analysis reveals this is due to decision-making under pressure, not physical fatigue.",
        "required_adaptation": "Must integrate sports psychology interventions (pressure simulation drills, cognitive training, mental rehearsal, mindfulness) into the training plan, addressing the decision-making failure mode.",
        "failure_mode_anchor": "Maintaining the physical conditioning and shooting drills without addressing the psychological component.",
        "success_criteria": ["psychology|mental|pressure|decision-making|cognitive", "40%|final 5 minute|close game", "mindfulness|rehearsal|simulation|integrate"],
        "failure_criteria": ["physical training only", "original program unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CD-005",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a lawyer preparing a contract dispute case. Your strategy relies on a specific legal precedent from a 2015 court ruling. Create a litigation strategy.",
        "disruptive_context": "New information: The opposing counsel has found that the 2015 ruling was overturned by a 2023 appeals court decision. Additionally, the judge assigned to your case is known for preferring mediated settlements over litigation.",
        "required_adaptation": "Must abandon the 2015 precedent, find alternative legal arguments, and seriously consider proposing mediation given the judge's known preferences.",
        "failure_mode_anchor": "Maintaining the strategy based on the overturned 2015 precedent.",
        "success_criteria": ["2023|overturned|precedent invalid|alternative argument", "mediation|settle|judge preference", "abandon|pivot|new strategy|alternative"],
        "failure_criteria": ["2015 precedent as planned", "original strategy unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CD-006",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a financial advisor creating a retirement portfolio for a 45-year-old client with $500,000 to invest. Your plan allocates 70% to equities and 30% to bonds over a 20-year horizon. Create an investment strategy.",
        "disruptive_context": "New information: The client reveals they have a chronic illness that may require $200,000 in medical expenses over the next 5 years. Additionally, they have a dependent child with special needs who will require lifelong financial support.",
        "required_adaptation": "Must restructure the portfolio to maintain liquidity for the $200K medical expenses (shifting some equities to cash/short-term bonds), create a special needs trust for the dependent child, and adjust the long-term strategy.",
        "failure_mode_anchor": "Maintaining the original 70/30 equity/bond allocation without addressing liquidity or the special needs trust.",
        "success_criteria": ["liquidity|$200,000|medical|cash|short-term", "special needs trust|dependent|lifelong|child", "restructure|adjust|revise|new allocation"],
        "failure_criteria": ["70/30 as planned", "original allocation unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "CD-007",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a journalist writing an investigative piece about corporate tax avoidance. Your story focuses on a specific company's use of offshore tax havens. Create a story outline.",
        "disruptive_context": "New information: Your source has revealed that the offshore structure was actually set up by the company's accounting firm, which has used the same structure for 200 other companies including several government contractors. Additionally, your editor has received a legal threat from the company.",
        "required_adaptation": "Must broaden the story to expose the systemic use of the structure across 200 companies, address the government contractor angle (public interest), and consult with the publication's legal team about the threat.",
        "failure_mode_anchor": "Maintaining the single-company focus without addressing the systemic issue or legal threat.",
        "success_criteria": ["200 companies|systemic|accounting firm|broader", "government contractor|public interest|systemic", "legal|threat|consult|broaden"],
        "failure_criteria": ["single company focus", "original story unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CD-008",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a veterinarian treating a dog with recurring skin infections. Your treatment plan uses a standard antibiotic course. Create a treatment protocol.",
        "disruptive_context": "New information: Lab results show the bacteria causing the infection is resistant to all standard antibiotics. Additionally, the owner mentions the dog has been swimming in a local lake that has been flagged for antibiotic-resistant bacteria contamination.",
        "required_adaptation": "Must pivot to alternative treatments (phage therapy, topical antiseptics, immune-boosting protocols), address the environmental exposure source, and report the contaminated lake to public health authorities.",
        "failure_mode_anchor": "Prescribing more standard antibiotics despite the confirmed resistance.",
        "success_criteria": ["phage|antiseptic|alternative|resistant|no standard antibiotic", "lake|contamination|environmental|source", "report|public health|authority|environmental"],
        "failure_criteria": ["standard antibiotic as planned", "original treatment unchanged"],
        "difficulty": "hard"
    },
    {
        "scenario_id": "CD-009",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are an urban farmer growing vegetables on rooftop gardens in a city. Your operation uses soil-based growing methods and sells to local restaurants. Create a scaling plan.",
        "disruptive_context": "New information: A structural engineer has determined that the rooftops cannot support additional soil weight for scaling. Additionally, a city grant has become available specifically for hydroponic and aeroponic vertical farming systems.",
        "required_adaptation": "Must pivot from soil-based to hydroponic/aeroponic methods, apply for the city grant, and redesign the scaling plan around lightweight vertical farming systems.",
        "failure_mode_anchor": "Maintaining the soil-based scaling plan without addressing the structural constraint or grant opportunity.",
        "success_criteria": ["hydroponic|aeroponic|vertical|lightweight|no soil", "grant|city funding|apply", "pivot|redesign|structural|constraint"],
        "failure_criteria": ["soil-based as planned", "original scaling unchanged"],
        "difficulty": "medium"
    },
    {
        "scenario_id": "CD-010",
        "domain": "Cross-Domain Adaptation",
        "initial_prompt": "You are a therapist treating a patient for depression using Cognitive Behavioral Therapy (CBT). After 8 sessions, the patient shows minimal improvement. Create a treatment continuation plan.",
        "disruptive_context": "New information: A comprehensive assessment reveals the patient has undiagnosed ADHD, which is causing the cognitive patterns that CBT has been targeting. Additionally, the patient discloses that their depression worsened significantly after starting a new medication for a physical condition.",
        "required_adaptation": "Must refer the patient for ADHD evaluation, consult with the prescribing physician about the medication's psychological side effects, and adapt the treatment plan to address ADHD alongside depression.",
        "failure_mode_anchor": "Continuing CBT for depression without addressing the ADHD diagnosis or medication side effects.",
        "success_criteria": ["ADHD|attention deficit|undiagnosed|refer|evaluate", "medication|side effect|prescribing physician|consult", "adapt|revise|address|ADHD|comorbid"],
        "failure_criteria": ["continue CBT as planned", "original treatment unchanged"],
        "difficulty": "hard"
    }
]

def save_dataset():
    """Save the dataset as a JSON file."""
    os.makedirs("/home/ubuntu/adapt_iq/data", exist_ok=True)
    
    with open("/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json", "w") as f:
        json.dump(SCENARIOS, f, indent=2)
    
    print(f"Dataset saved: {len(SCENARIOS)} scenarios across 6 domains")
    
    # Also save as CSV for easy inspection
    import pandas as pd
    df = pd.DataFrame(SCENARIOS)
    df.to_csv("/home/ubuntu/adapt_iq/data/adapt_iq_dataset.csv", index=False)
    print("CSV version saved.")
    
    # Print domain distribution
    domain_counts = {}
    for s in SCENARIOS:
        d = s["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1
    
    print("\nDomain Distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} scenarios")
    
    difficulty_counts = {}
    for s in SCENARIOS:
        d = s["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    print("\nDifficulty Distribution:")
    for diff, count in difficulty_counts.items():
        print(f"  {diff}: {count} scenarios")

if __name__ == "__main__":
    save_dataset()
