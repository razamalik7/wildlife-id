# Reference Photo Sources — Survey for the AnimlVision Art Pipeline

*Researched 2026-08-11. Goal: one commercially-safe, field-guide-pose reference photo per species for ~1,400 US vertebrates (birds, mammals, reptiles, amphibians), picked automatically.*

**Hard requirements:** programmatic API, per-photo license metadata, CC0 / CC-BY / public domain only (no NC, no SA), and realistic odds of a single-adult / side-or-three-quarter / whole-body / sharp photo — for obscure species (Idaho Giant Salamander), not just charismatic ones (Bald Eagle).

**License note for this use case:** CC-BY and CC0 explicitly permit commercial derivative works (the stylized illustration is a derivative; keep attribution records for CC-BY). "Free" stock licenses (Unsplash/Pexels/Pixabay) are *not* CC — they carry platform-specific anti-scraping and anti-ML clauses that make them legally worse than CC-BY despite being "free." Details per source below.

---

## Tier 1 — Primary candidates

### 1. iNaturalist (API + AWS Open Data)

The single deepest source of live-animal, in-habitat photos of North American vertebrates, with per-photo machine-readable licenses. The REST API (`api.inaturalist.org/v1`, [docs](https://api.inaturalist.org/v2/docs/), [developers page](https://www.inaturalist.org/pages/developers)) supports `GET /observations?taxon_name=...&photo_license=cc0,cc-by&quality_grade=research`, returning `license_code` on every photo plus community-vetted IDs (`quality_grade=research`), fave counts for ranking, and taxon endpoints with curated `default_photo`s. **License situation:** per-photo CC license chosen by each observer; the default is CC-BY-NC, so the commercially-safe pool is roughly 10–20% of photos — filter with `photo_license=cc0,cc-by`. Photos with open licenses are served from `inaturalist-open-data.s3.amazonaws.com`; all-rights-reserved photos live on `static.inaturalist.org`, a clean structural signal ([open-data docs](https://github.com/inaturalist/inaturalist-open-data)). **Bulk access:** the [iNaturalist Licensed Observation Images AWS Open Data bucket](https://registry.opendata.aws/inaturalist-open-data/) (`s3://inaturalist-open-data`, no AWS account needed, `--no-sign-request`) ships monthly TSV snapshots (`photos.csv.gz`, `observations.csv.gz`, `taxa.csv.gz`, `observers.csv.gz`) including a license column per photo — you can do the entire 1,400-species selection offline with zero rate-limit exposure, then download `photos/[photo_id]/original.jpg` from S3 ([blog announcement](https://www.inaturalist.org/blog/49564-inaturalist-licensed-observation-images-in-the-amazon-open-data-sponsorship-program)). **Rate limits (API path):** throttled at 100 req/min, asked to stay ≤60 req/min and <10,000 req/day ([recommended practices](https://www.inaturalist.org/pages/api+recommended+practices)) — 1,400 species × a few pages each fits in a day, but the S3 route is better. **Coverage spot-check:** *Dicamptodon aterrimus* (Idaho Giant Salamander) returns 11 research-grade observations with CC0/CC-BY photos via the live API (many more under NC) — thin but non-zero for one of the most obscure US herps; common species return thousands. **Quality/curation:** uncurated phone photos through DSLR work; pose quality varies wildly, which is exactly what the existing `photo_funnel.py` scorer is for. Fave counts and multiple candidate photos per species give the funnel plenty to chew on. **Cost:** free. **Verdict: primary source.**

### 2. Wikimedia Commons (+ Quality/Valued Images, + Wikidata P18)

The best *curated* free-license source. Access via the MediaWiki Action API (`commons.wikimedia.org/w/api.php`, [Commons:API](https://commons.wikimedia.org/wiki/Commons:API)) — full-text/category search plus `imageinfo&iiprop=extmetadata`, which returns structured `LicenseShortName`, `LicenseUrl`, `UsageTerms`, `AttributionRequired` per file ([API:Imageinfo](https://www.mediawiki.org/wiki/API:Imageinfo), [Commons:Machine-readable data](https://commons.wikimedia.org/wiki/Commons:Machine-readable_data)). Everything on Commons is free-licensed by policy, but **licenses include SA and GFDL variants — you must still filter extmetadata per file** for CC0/CC-BY/PD. Curation signals are unmatched: search `incategory:Featured_pictures_on_Wikimedia_Commons|Quality_images|Valued_images` to restrict to human-reviewed images, and the CommonsMetadata extension exposes `Assessments` values (`featured`, `quality`, `valued`, `poty`, `potd`) ([Extension:CommonsMetadata](https://www.mediawiki.org/wiki/Extension:CommonsMetadata)). Valued Images in particular are chosen as *the most useful illustration of a subject* — very close to "field-guide pose." **Wikidata P18:** nearly every taxon item has a hand-picked portrait image (`?taxon wdt:P225 "Dicamptodon aterrimus"; wdt:P18 ?image`) queryable in one SPARQL batch for all 1,400 species at `query.wikidata.org` ([P18 usage](https://www.wikidata.org/wiki/Property_talk:P18), [SPARQL examples](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples)) — a free, pre-curated "one good image per species" table, though P18 only returns best-rank images. **Coverage:** excellent for birds/mammals; thinner and noisier for obscure herps — a live search for "Idaho giant salamander" returned 21 file hits, several of which are scanned book plates rather than photos, so automated filtering is required. **Rate limits:** generous (etiquette-based; parallel-request limits, no hard daily cap for read queries). **Cost:** free. **Verdict: co-primary for curated picks; the P18 SPARQL batch is the cheapest first pass in the whole stack.**

### 3. Flickr API (CC-filtered)

The deepest pool of *high-production-value* wildlife photography with per-photo license codes. `flickr.photos.search` accepts a `license` parameter; the safe IDs are **4 (CC-BY 2.0), 11 (CC-BY 4.0), 9 (CC0), 10 (Public Domain Mark), 8 (US Government Work)** — avoid 1/2/3/5/6/12+ (NC/ND/SA variants) ([license table](https://www.flickr.com/services/api/flickr.photos.licenses.getInfo.html)). Many serious bird/herp photographers post CC-BY at resolutions and sharpness iNat rarely matches, and USFWS/NPS/USGS official accounts post PD images here. **Weaknesses:** species labeling is free-text tags/titles — no taxonomy, so *Sceloporus* confusions and outright misidentifications are common; every candidate must be verified (your CLIP/judge funnel, or cross-check by searching scientific name exactly). **Rate limits:** 3,600 queries/hour per key ([developer guide](https://www.flickr.com/services/developer/api/)); ~1,400 species × 2–3 searches fits in ~1 hour. **License/ToS caveats:** a commercial API key is required if the app's *primary purpose* is revenue; you're solely responsible for complying with each CC license (attribution for CC-BY); caching restrictions apply to photos but a one-time download of a CC-BY file under its CC license is use of the license, not the API cache ([API ToS](https://www.flickr.com/services/api/tos/)). **Cost:** free (commercial keys negotiated). **Verdict: strongest fallback for photo quality; requires ID verification.**

---

## Tier 2 — Aggregators and government PD sources

### 4. GBIF occurrence/media API

GBIF is the aggregator of record: `api.gbif.org/v1/occurrence/search?scientificName=...&mediaType=StillImage` returns occurrences with a `media[]` array carrying `identifier` (direct image URL), `license`, `creator`, and `publisher` ([API reference](https://techdocs.gbif.org/en/openapi/), [multimedia publishing docs](https://techdocs.gbif.org/en/data-publishing/multimedia-publishing), [multimedia blog](https://data-blog.gbif.org/post/gbif-multimedia/)). Its enormous value is *taxonomic normalization* (name matching via the species API) and one query surface over iNaturalist, Observation.org, museum collections, and university datasets. **Critical caveat:** media items may be licensed more restrictively than the occurrence record — always read the license on the media object itself, never the dataset license ([GBIF multimedia docs](https://techdocs.gbif.org/en/data-publishing/multimedia-publishing)). Live spot-check for *Dicamptodon aterrimus* returned image-bearing occurrences whose photos are iNaturalist S3 URLs, mostly CC-BY-NC — i.e., **for US vertebrates GBIF's image pool is largely iNat re-syndicated**, so as a *photo* source it adds little over going to iNat directly; its occurrence API also can't filter *by media license*. There is also an experimental [occurrence image cache API](https://techdocs.gbif.org/en/openapi/images) for thumbnails. **Rate limits:** generous, anonymous access fine for this scale; free. **Verdict: use for taxonomy backbone + discovering non-iNat datasets with open media; not a primary photo source.**

### 5. Smithsonian Open Access

4.5M+ media assets released **CC0** — the cleanest license in this survey ([Open Access FAQ](https://www.si.edu/openaccess/faq), [CC announcement](https://creativecommons.org/2020/02/27/smithsonian-releases-2-8-million-images-data-into-the-public-domain-using-cc0/)). API: `GET https://api.si.edu/openaccess/api/v1.0/search?q=...&api_key=KEY` with a free api.data.gov key; 1,000 req/hour registered (30/hour on DEMO_KEY) ([api.data.gov rate limits](https://api.data.gov/docs/rate-limits/)). Bulk metadata lives in an AWS Open Data bucket ([registry](https://registry.opendata.aws/smithsonian-open-access/), [GitHub, now archived to AWS](https://github.com/Smithsonian/OpenAccess)). **Coverage reality:** the wildlife-relevant material is mostly NMNH *specimen* photographs (study skins, fluid-preserved herps) and National Zoo photos — museum drawer shots, not field-guide poses. Occasional live-animal Zoo portraits are superb but cover only exhibit species. **Verdict: niche fallback; CC0 makes anything it does have frictionless.**

### 6. USFWS National Digital Library

Public-domain wildlife photos by agency photographers — "free to be used as you wish," credit requested ([FWS library page](https://www.fws.gov/library), [collection description](https://digitalibra.omeka.net/items/show/57), [example record](https://digitalmedia.fws.gov/digital/collection/natdiglib/id/33782/)). Runs on CONTENTdm (`digitalmedia.fws.gov`), which exposes the stock CONTENTdm `dmwebservices` JSON API (dmQuery/dmGetItemInfo) — functional but clunky, with no taxonomy (free-text titles/subjects). **Coverage:** strong for refuge-system birds, waterfowl, and listed species (often excellent field-guide poses by staff photographers); near-zero for obscure salamanders and small mammals outside the refuge system. Note many USFWS PD images are *also* on Flickr ([USFWS account](https://www.flickr.com/photos/usfwshq/)) and Commons, where they're easier to query. **Cost/limits:** free, no published limits. **Verdict: supplementary PD source, best reached indirectly via Flickr license=8 or Commons.**

### 7. NPS API (multimedia)

Developer-friendly REST API at `developer.nps.gov` with a free api.data.gov key, 1,000 req/hour default ([NPS data API page](https://www.nps.gov/subjects/digital/nps-data-api.htm), [api.data.gov limits](https://api.data.gov/docs/rate-limits/)); ~29 endpoints including `multimedia/galleries` and galleries/assets ([API guides](https://www.nps.gov/subjects/developer/guides.htm)). NPS-produced media is US Government work → public domain. **Coverage:** organized around parks and interpretation, not taxonomy — you can't ask it for a species list, and species depth is shallow (charismatic megafauna, some birds). **Verdict: marginal for this pipeline; skip except as a curiosity-driven gap filler.**

### 8. Encyclopedia of Life (EOL)

Aggregator of Commons + Flickr + iNat + museum images keyed to taxon pages. Classic API: `https://eol.org/api/pages/1.0/{page_id}.json?details=true&images_per_page=75&licenses=pd|cc-by` returns dataObjects with per-image `license`, `rightsHolder`, and `eolMediaURL` ([classic APIs doc](https://eol.org/docs/what-is-eol/classic-apis), [data services](https://eol.org/docs/what-is-eol/data-services)). All EOL content is PD or CC by policy ([data services](https://eol.org/docs/what-is-eol/data-services)). **The catch:** the classic API is aging and flaky — community reports of 500 errors when combining `licenses` filters with some taxa, and periodic outages ([license-filter bug thread](https://discuss.eol.org/t/api-error-when-specifying-licenses/2441), [downtime thread](https://discuss.eol.org/t/eol-classic-api-down/2416)). Media it serves for US vertebrates is overwhelmingly the same Commons/Flickr/iNat photos you can get first-hand with better metadata. **Verdict: convenient taxon-keyed index, but a re-syndicator with reliability risk — use only if you want one API instead of three, and dedupe against primaries.**

### 9. Openverse API

WordPress-run search over ~800M openly-licensed items indexed from Flickr, Commons, museums, etc. `api.openverse.org` `/v1/images/` supports `license=cc0,by,pdm` and `license_type=commercial` filters with per-result license metadata ([API listing](https://apis.io/apis/openverse/openverse-api/), [JS client docs](https://docs.openverse.org/packages/js/api_client/index.html)). **Limits:** anonymous access is a stingy 5 req/hour / 100 req/day; registered OAuth2 apps get substantially more ([API listing](https://apis.io/apis/openverse/openverse-api/)). Index can lag or mis-cache license state, so verify at the source record before shipping. No taxonomy; keyword search only. **Verdict: handy unified fallback search (one query hits Flickr+Commons at once); not a system of record.**

---

## Tier 3 — Ruled out (license or API failures)

### 10. Macaulay Library / eBird — **unusable commercially**

Best bird photo archive on Earth, rank-sorted by community ratings — and effectively closed. Contributors grant Cornell a license for its research/educational mission; **commercial use requires a negotiated license agreement per use** ([media licensing agreement](https://support.ebird.org/en/support/solutions/articles/48000952192-cornell-media-licensing-agreement), [request/download policy](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media)). The eBird API is *data-only* (no media) and its ToS forbids any commercial use without written permission ([eBird API terms](https://www.birds.cornell.edu/home/ebird-api-terms-of-use/)). Since 2021 contributors *can* opt individual media into CC-BY/CC-BY-SA/CC0 ([announcement](https://ebird.org/news/creative-commons-license-for-ebird-media)), but there is no public API to search by that flag — access still goes through the manual request pipeline. **Verdict: excluded. At 1,400-species scale, per-item negotiation is a non-starter.**

### 11. Observation.org — **effectively unusable**

Large European recording platform with growing US data. Has a real REST API (OAuth2, registration required) ([API docs](https://waarneming.nl/api/docs/)), but photos carry heterogeneous licenses with no unified open grant and no per-photo open-license search filter ([API docs](https://waarneming.nl/api/docs/), [Commons category for the small relicensed subset](https://commons.wikimedia.org/wiki/Category:Images_from_Waarneming.nl)); rights stay with observers. Its GBIF exports are data-oriented; images that surface there must still be license-checked individually. US coverage is also far thinner than iNat's. **Verdict: excluded — any usable open photos from it will surface via GBIF media-license checks anyway.**

### 12. CalPhotos — **unusable commercially**

~800k organism photos with genuinely good herp/small-mammal depth and a simple XML/URL query interface — but licensing kills it: photographers retain copyright, most images are "educational/non-commercial with attribution," and **commercial use requires contacting each photographer via a permission form, often with fees** ([licenses page](https://calphotos.berkeley.edu/licenses.html), [FAQ](https://calphotos.berkeley.edu/faq.html)). A minority are CC-licensed and license is shown per photo, but there's no reliable API-level CC-only commercial filter to build on. **Verdict: excluded as a pipeline source; acceptable as a manual last-resort lookup for a handful of holdout species *if* an individual photo is clearly CC-BY.**

### 13. Animal Diversity Web — **unusable commercially**

University of Michigan's taxon encyclopedia. Site text is CC-BY-NC-SA; images are a per-item mix of CC (mostly NC variants), PD, and all-rights-reserved, with permission requests directed to individual copyright holders ([conditions of use](https://animaldiversity.org/about/use_conditions/)). No media API. **Verdict: excluded.**

### 14. HerpMapper — **unusable**

The closest thing to a "BugGuide for herps" with vouchered photos of exactly the obscure US species this project needs — but content may only be downloaded for **personal, non-commercial use**; commercial use requires written authorization, and data access is gated to conservation/research partners ([terms via partner sites](https://www.sdherps.org/terms), [partners page](https://herpmapper.org/partners), [FAQ](https://www.herpmapper.org/faq)). No public media API. **Verdict: excluded. (For herp depth, iNaturalist is the open-licensed equivalent — its herp community is enormous.)**

### 15. Unsplash — **unusable for this pipeline**

License permits commercial use without attribution, but prohibits "compiling images... to replicate a similar or competing service" — explicitly including mass scraping ([license](https://unsplash.com/license), [help article](https://help.unsplash.com/en/articles/2612332-what-do-you-mean-by-compiling-images-to-replicate-a-similar-or-competing-service)) — and Unsplash's own ML dataset is licensed **non-commercial only**, with a stated position against commercial ML use of their imagery ([datasets TERMS](https://github.com/unsplash/datasets/blob/master/TERMS.md), [API terms](https://unsplash.com/api-terms)). API is fine (50 req/hr demo, ~1,000–5,000/hr production, free; hotlinking + attribution required by API guidelines — [docs](https://unsplash.com/documentation)). **Coverage:** effectively zero for pocket mice, plethodontid salamanders, or most sparrows — keyword-tagged aesthetic photography, no taxonomy, heavy duplication of ~50 charismatic species. **Verdict: excluded — coverage fails even before the license analysis does.**

### 16. Pexels — **unusable for this pipeline**

Free commercial use with modification allowed ([license](https://www.pexels.com/license/)); free API (200 req/hr, 20k/month class limits). But the license/ToS framework was written for design assets, prohibits redistribution on competing platforms, and its API terms don't affirmatively permit bulk ML/derivative pipelines; species coverage has the same "50 charismatic species" profile as Unsplash with unreliable labels. **Verdict: excluded on coverage; license ambiguity for AI-derivative use is a second strike ([license traps analysis](https://www.licenseorg.com/blog/free-stock-photos-licensing-traps)).**

### 17. Pixabay — **unusable**

The current Pixabay Content License **expressly prohibits use of content for machine-learning training and automated scraping/extraction for ML purposes** ([terms](https://pixabay.com/service/terms/), [license summary](https://pixabay.com/service/license-summary/), [risk analysis](https://picdefense.io/resources/source-intel/pixabay/)). Since this pipeline conditions image generation on the photo, this is squarely in the prohibited zone. **Verdict: excluded outright.**

### 18. iDigBio / VertNet (specimen aggregators) — honorable mention

`search.idigbio.org/v2/search/records/` plus a media API with per-media-record metadata ([media API demo](https://idigbio.github.io/ridigbio/articles/MediaAPIDemo.html), [code samples](https://github.com/iDigBio/idigbio-search-api/wiki/Code-Samples)). Note iDigBio data stopped updating May 2026 with services migrating toward a GBIF hosted portal ([portal notice](https://portal.idigbio.org/)). Content is museum *specimen* imagery — study skins and preserved herps, wrong pose class for a field guide. **Verdict: excluded for reference photos; possibly useful someday for scale/anatomy cross-checks.**

---

## Live coverage spot-check (Idaho Giant Salamander, *Dicamptodon aterrimus*)

| Source | Result (2026-08-11) |
|---|---|
| iNaturalist API, `photo_license=cc0,cc-by`, research-grade | **11 observations** with commercially-safe photos (first hit CC-BY, S3-hosted) |
| GBIF `mediaType=StillImage` | Image-bearing occurrences are iNat re-syndications, mostly CC-BY-NC — confirms GBIF adds little beyond iNat for US herps |
| Wikimedia Commons file search | 21 hits for "Idaho giant salamander," several are scanned book plates → needs photo-vs-scan filtering |
| Unsplash/Pexels | No usable species-verified matches (keyword search, no taxonomy) |

Takeaway: for the long tail, **only iNaturalist reliably has commercially-safe live-animal photos**, and even there the open-licensed subset is ~an order of magnitude smaller than the NC pool. Expect a manual-rescue bucket of perhaps 2–5% of species where no CC0/CC-BY photo passes the pose funnel.

---

## Recommended stack

**Stage 0 — Wikidata P18 batch (1 SPARQL query).** Pull the curated portrait filename for all 1,400 taxa from `query.wikidata.org`, resolve via Commons `imageinfo&iiprop=extmetadata`, keep only CC0/CC-BY/PD. Instant, human-curated baseline; expect ~60–80% of species to get a *candidate* (not always a great pose).

**Stage 1 — iNaturalist bulk (primary).** Sync `photos.csv.gz` + `observations.csv.gz` + `taxa.csv.gz` from `s3://inaturalist-open-data` (no key, no rate limit), join to the species list by taxon, filter license ∈ {CC0, CC-BY} and quality_grade=research, rank by faves/recency/resolution, download top-N `original.jpg` per species from S3, and run them through `photo_funnel.py` for pose scoring. This is the workhorse: deepest long-tail coverage, per-photo licenses, zero API friction.

**Stage 2 — Commons Quality/Valued Images (curation upgrade).** For each species, query Commons `incategory:Quality_images|Valued_images|Featured_pictures_on_Wikimedia_Commons` + taxon name; where a QI/VI/FP beats the iNat winner in funnel score, take it (license-check via extmetadata; skip SA/GFDL).

**Stage 3 — Flickr CC sweep (quality fallback).** For species whose best candidate scores below the funnel threshold, `flickr.photos.search` with `license=4,9,10,11,8`, scientific + common name, `sort=interestingness-desc`; verify species ID with the funnel's judge before accepting (Flickr has no taxonomy).

**Stage 4 — Gap fillers.** Smithsonian OA (CC0, api.data.gov key) and USFWS-via-Flickr (`license=8`) for remaining holes; GBIF media search to catch any non-iNat open-licensed museum/agency photos (check the *media-object* license, not the dataset license); Openverse as a one-query cross-source sweep.

**Stage 5 — Manual rescue (~2–5% of species).** Human pick from CalPhotos CC-BY items, direct photographer contact, or commission. Track in `vetting/candidates.json` like the rest.

**Excluded on license:** Macaulay/eBird (commercial requires negotiated license), Pixabay (ML use prohibited), Unsplash/Pexels (anti-compilation/ML clauses + no long-tail coverage), CalPhotos/ADW/HerpMapper (NC or per-photographer permission), Observation.org (no open-license API filter).

**Pipeline hygiene:** record `{photo_id, source, url, license, license_url, attribution}` at fetch time (iNat users can change licenses later — the snapshot is your evidence); dedupe across sources by iNat photo ID / perceptual hash (Commons, GBIF, EOL, and Openverse all re-serve iNat and Flickr content); CC-BY attribution must survive into the shipped product's credits.
