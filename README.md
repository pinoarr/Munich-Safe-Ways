# About the Project

Munich Safe Ways began with a simple frustration about everyday navigation. Standard maps treat the city as an abstract graph in which only distance matters. Yet anyone who walks or cycles through Munich knows that two routes of identical length can feel radically different. One may cut through a loud traffic corridor, the other winds past trees and cafés. One shortcut might be technically efficient, the other feels empty or poorly lit after dark. We wanted a routing engine that captures these experiential differences and reflects the human reality of moving through the city, not just its geometry.

## Connecting the Idea to Munich

From the start we set out to *tailor* the concept directly to Munich. The city has a distinctive texture: continuous green spaces, a rich network of quiet residential streets, numerous parks and riverbanks, the Isar as a linear green spine, and a transit system that shapes where people naturally gather. We examined this structure through OpenStreetMap and Munich’s open datasets, which provided detailed information on parks, waterways, transit stops, amenities and urban form. Our aim was to translate these elements into computable features that express comfort, liveliness and perceived safety.

## Three Specialised Routing Modes

We implemented three modes designed for common real-world situations in Munich.

**Direct mode** mimics the impatient commuter that only cares about getting there quickly. It intentionally ignores greenery, penalises well-lit streets and even favours gritty shortcuts so the other modes have a clear contrast.

**Night walking safety** assesses isolation, lighting, activity anchors and our imported list of historical crime incidents (based on fuchsvomwalde/munichwatcher) to guide users toward populated, better lit streets.

**Scenic mode** highlights streets lined with greenery and water and favours open and quiet pathways that provide a sense of calm.

Every response also reports distance, estimated travel time and a bundle of contextual metrics (average scenic score, lighting coverage, proximity to help, exposure to hotspots). The UI compares these numbers against the Direct baseline so you can see statements such as “+45% more greenery vs Direct” or “–30% crime hotspots vs Direct” right after selecting a route.

For every street segment ( e ) we assemble a feature vector representing these properties and define
$$
\text{cost}(e) = d(e) + \lambda, r(e),
$$
where \\( r(e) \\) encodes discomfort or risk and \\( \lambda \\) controls how strongly a user values safety or comfort over speed. A slider allows people to move smoothly along the spectrum between efficiency and well-being.

## What We Want to Learn

We want to explore how far deterministic modelling can go when the underlying data has strong spatial structure. We also wanted to understand how citizens might benefit when navigation aligns with how humans perceive streets: through light, noise, greenery, openness and social presence. Working with OSM together with Munich’s open data layers demonstrated how expressive these sources become when combined. Computing street level comfort and safety pushed us toward careful reasoning about geometry, proximity and scale, which deepened our understanding of urban data.

## Why This Can Improve Daily Life

We believe Munich Safe Ways can genuinely improve everyday mobility. Cyclists can avoid stressful junctions. Walkers can discover calmer and more pleasant paths through green corridors. People returning home at night can choose routes that feel more visible and socially connected. By grounding the routing logic in the actual spatial structure of Munich and keeping the model transparent and interpretable, we offer a navigation tool that respects not only where people want to go, but also how they want to feel along the way.
