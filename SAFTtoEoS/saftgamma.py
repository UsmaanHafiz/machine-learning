





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-TD/1gdStKo77Iu1A+WNb+2YoRJv7l1hZVO0/0h+eJ5wyGxiTvIbLRko1dQ1jEa9w5BK3poi18st9OPel71zK8A==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-94d2de65a56b07e193e50e91873678b6.css" />
  
    <link crossorigin="anonymous" media="all" integrity="sha512-s/fszKZ5jV9+tAHibuwhvymgy78cB49lbU1L/E40XqZNE6sBv07dSOCEz7I1n45OD2XXg8mUCI2Ub+pR6AegmA==" rel="stylesheet" href="https://github.githubassets.com/assets/github-3c4881237e97cb689271ee2f91e36269.css" />
    
    
    
    

  <meta name="viewport" content="width=device-width">
  
  <title>ad-saftgamma/saftgamma.py at master · kezhengzhu/ad-saftgamma</title>
    <meta name="description" content="Contribute to kezhengzhu/ad-saftgamma development by creating an account on GitHub.">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    <meta name="twitter:image:src" content="https://avatars3.githubusercontent.com/u/44840976?s=400&amp;v=4" /><meta name="twitter:site" content="@github" /><meta name="twitter:card" content="summary" /><meta name="twitter:title" content="kezhengzhu/ad-saftgamma" /><meta name="twitter:description" content="Contribute to kezhengzhu/ad-saftgamma development by creating an account on GitHub." />
    <meta property="og:image" content="https://avatars3.githubusercontent.com/u/44840976?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="kezhengzhu/ad-saftgamma" /><meta property="og:url" content="https://github.com/kezhengzhu/ad-saftgamma" /><meta property="og:description" content="Contribute to kezhengzhu/ad-saftgamma development by creating an account on GitHub." />

  <link rel="assets" href="https://github.githubassets.com/">
  <link rel="web-socket" href="wss://live.github.com/_sockets/VjI6NDUxODA1MTI3OjNjMWE1MjM5NjU3OTdjN2YxNmIyYjcyYzM4ZDc5MGRmNTAyYTkyMjdlZTQ5N2ZjYjI3MDBmZWQzYWJmZTVhMTI=--f76d03f32532949cee36822e1c5e00c6f06bf575">
  <link rel="sudo-modal" href="/sessions/sudo_modal">
  <meta name="request-id" content="CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7" data-pjax-transient>


  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

      <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
    <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
    <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

  <meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-request_id" content="CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7" /><meta name="octolytics-dimension-region_edge" content="ams" /><meta name="octolytics-dimension-region_render" content="iad" /><meta name="octolytics-dimension-ga_id" content="" class="js-octo-ga-id" /><meta name="octolytics-dimension-visitor_id" content="8743668296473420040" /><meta name="octolytics-actor-id" content="56049634" /><meta name="octolytics-actor-login" content="UsmaanHafiz" /><meta name="octolytics-actor-hash" content="6e2154c25cc11cd6910cf79ace1c260f4a92a60a0c1de9e19ec1c3f05fd00fab" />
<meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />



    <meta name="google-analytics" content="UA-3769691-2">

  <meta class="js-ga-set" name="userId" content="a48024111ae10e3f753bdf142e400d6c">

<meta class="js-ga-set" name="dimension1" content="Logged In">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="UsmaanHafiz">

      <meta name="expected-hostname" content="github.com">
    <meta name="js-proxy-site-detection-payload" content="MDMwNjk4MGQyZjEyM2Q4OWI3M2VhNTE4MWY5ZTZkMjdkNTRjNTFhNjNkNDM5OGM2NjYxNDIxMjg2ZDg0MzkxNnx7InJlbW90ZV9hZGRyZXNzIjoiODIuMjYuMjEwLjU2IiwicmVxdWVzdF9pZCI6IkNGNEQ6MUFCNkM6MkYwMkM5NDo0NUVBQjEzOjVEQjVDQkU3IiwidGltZXN0YW1wIjoxNTcyMTk1MzExLCJob3N0IjoiZ2l0aHViLmNvbSJ9">

    <meta name="enabled-features" content="ACTIONS_V2_ON_MARKETPLACE,MARKETPLACE_FEATURED_BLOG_POSTS,MARKETPLACE_INVOICED_BILLING,MARKETPLACE_SOCIAL_PROOF_CUSTOMERS,MARKETPLACE_TRENDING_SOCIAL_PROOF,MARKETPLACE_RECOMMENDATIONS,MARKETPLACE_PENDING_INSTALLATIONS,NOTIFY_ON_BLOCK,RELATED_ISSUES,GHE_CLOUD_TRIAL">

  <meta name="html-safe-nonce" content="f93c5c570fc68442cbae4d2c00fff3f14ba9a8b6">

  <meta http-equiv="x-pjax-version" content="eebadb79980e31602c9071d050b11f46">
  

      <link href="https://github.com/kezhengzhu/ad-saftgamma/commits/master.atom" rel="alternate" title="Recent Commits to ad-saftgamma:master" type="application/atom+xml">

  <meta name="go-import" content="github.com/kezhengzhu/ad-saftgamma git https://github.com/kezhengzhu/ad-saftgamma.git">

  <meta name="octolytics-dimension-user_id" content="44840976" /><meta name="octolytics-dimension-user_login" content="kezhengzhu" /><meta name="octolytics-dimension-repository_id" content="208090094" /><meta name="octolytics-dimension-repository_nwo" content="kezhengzhu/ad-saftgamma" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="208090094" /><meta name="octolytics-dimension-repository_network_root_nwo" content="kezhengzhu/ad-saftgamma" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://github.githubassets.com/favicon.ico">

<meta name="theme-color" content="#1e2327">



  <meta name="webauthn-auth-enabled" content="true">

  <meta name="webauthn-registration-enabled" content="true">

  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-in env-production page-responsive page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="p-3 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
    <span class="Progress progress-pjax-loader position-fixed width-full js-pjax-loader-bar">
      <span class="progress-pjax-loader-bar top-0 left-0" style="width: 0%;"></span>
    </span>

    
    
    


          <header class="Header js-details-container Details flex-wrap flex-lg-nowrap p-responsive" role="banner">

    <div class="Header-item d-none d-lg-flex">
      <a class="Header-link" href="https://github.com/" data-hotkey="g d" aria-label="Homepage" data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
</a>

    </div>

    <div class="Header-item d-lg-none">
      <button class="Header-link btn-link js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
        <svg height="24" class="octicon octicon-three-bars" viewBox="0 0 12 16" version="1.1" width="18" aria-hidden="true"><path fill-rule="evenodd" d="M11.41 9H.59C0 9 0 8.59 0 8c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zm0-4H.59C0 5 0 4.59 0 4c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zM.59 11H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1H.59C0 13 0 12.59 0 12c0-.59 0-1 .59-1z"/></svg>
      </button>
    </div>

    <div class="Header-item Header-item--full flex-column flex-lg-row width-full flex-order-2 flex-lg-order-none mr-0 mr-lg-3 mt-3 mt-lg-0 Details-content--hidden">
        <div class="header-search flex-self-stretch flex-lg-self-auto mr-0 mr-lg-3 mb-3 mb-lg-0 scoped-search site-scoped-search js-site-search position-relative js-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" role="search" aria-label="Site" data-scope-type="Repository" data-scope-id="208090094" data-scoped-search-url="/kezhengzhu/ad-saftgamma/search" data-unscoped-search-url="/search" action="/kezhengzhu/ad-saftgamma/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
      <label class="form-control input-sm header-search-wrapper p-0 header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control input-sm header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search or jump to…"
          data-unscoped-placeholder="Search or jump to…"
          data-scoped-placeholder="Search or jump to…"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search or jump to…"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations#csrf-token=x/G+vejvbBHYRMXlogFuRyECD4zEBnGJz0X8XTzKblojGbVRUmcyFIVrwGzugvWXWldnly8htTz4R9y+eobg8A=="
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 12 16" version="1.1" role="img"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 15 16" version="1.1" role="img"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M15.7 13.3l-3.81-3.83A5.93 5.93 0 0013 6c0-3.31-2.69-6-6-6S1 2.69 1 6s2.69 6 6 6c1.3 0 2.48-.41 3.47-1.11l3.83 3.81c.19.2.45.3.7.3.25 0 .52-.09.7-.3a.996.996 0 000-1.41v.01zM7 10.7c-2.59 0-4.7-2.11-4.7-4.7 0-2.59 2.11-4.7 4.7-4.7 2.59 0 4.7 2.11 4.7 4.7 0 2.59-2.11 4.7-4.7 4.7z"/></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


    <li class="d-flex flex-justify-center flex-items-center p-0 f5 js-jump-to-suggestion">
      <img src="https://github.githubassets.com/images/spinners/octocat-spinner-128.gif" alt="Octocat Spinner Icon" class="m-2" width="28">
    </li>
</ul>

            </div>
      </label>
</form>  </div>
</div>


      <nav class="d-flex flex-column flex-lg-row flex-self-stretch flex-lg-self-auto" aria-label="Global">
    <a class="Header-link d-block d-lg-none py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:dashboard:user" aria-label="Dashboard" href="/dashboard">
      Dashboard
</a>
  <a class="js-selected-navigation-item Header-link  mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15" data-hotkey="g p" data-ga-click="Header, click, Nav menu - item:pulls context:user" aria-label="Pull requests you created" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls" href="/pulls">
    Pull requests
</a>
  <a class="js-selected-navigation-item Header-link  mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15" data-hotkey="g i" data-ga-click="Header, click, Nav menu - item:issues context:user" aria-label="Issues you created" data-selected-links="/issues /issues/assigned /issues/mentioned /issues" href="/issues">
    Issues
</a>
    <div class="mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15">
      <a class="js-selected-navigation-item Header-link" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-octo-click="marketplace_click" data-octo-dimensions="location:nav_bar" data-selected-links=" /marketplace" href="/marketplace">
        Marketplace
</a>      

    </div>

  <a class="js-selected-navigation-item Header-link  mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore" href="/explore">
    Explore
</a>


    <a class="Header-link d-block d-lg-none mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15" href="https://github.com/UsmaanHafiz">
      <img class="avatar" height="20" width="20" alt="@UsmaanHafiz" src="https://avatars2.githubusercontent.com/u/56049634?s=60&amp;v=4" />
      UsmaanHafiz
</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="P78D6pwhZimwsJ8bPe4wmG8dr6d1esJK6igBUI+o4YRt80+q95MxCYFNrgETiUOALdrU3xDuABIS0YTxuEGInQ==" />
      <button type="submit" class="Header-link mr-0 mr-lg-3 py-2 py-lg-0 border-top border-lg-top-0 border-white-fade-15 d-lg-none btn-link d-block width-full text-left" data-ga-click="Header, sign out, icon:logout" style="padding-left: 2px;">
        <svg class="octicon octicon-sign-out v-align-middle" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 9V7H8V5h4V3l4 3-4 3zm-2 3H6V3L2 1h8v3h1V1c0-.55-.45-1-1-1H1C.45 0 0 .45 0 1v11.38c0 .39.22.73.55.91L6 16.01V13h4c.55 0 1-.45 1-1V8h-1v4z"/></svg>
        Sign out
      </button>
</form></nav>

    </div>

    <div class="Header-item Header-item--full flex-justify-center d-lg-none position-relative">
      <div class="css-truncate css-truncate-target width-fit position-absolute left-0 right-0 text-center">
              <svg class="octicon octicon-repo" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
    <a class="Header-link" href="/kezhengzhu">kezhengzhu</a>
    /
    <a class="Header-link" href="/kezhengzhu/ad-saftgamma">ad-saftgamma</a>

</div>
    </div>


    <div class="Header-item mr-0 mr-lg-3 flex-order-1 flex-lg-order-none">
      

    <a aria-label="You have no unread notifications" class="Header-link notification-indicator position-relative tooltipped tooltipped-s js-socket-channel js-notification-indicator" data-hotkey="g n" data-ga-click="Header, go to notifications, icon:read" data-channel="notification-changed:56049634" href="/notifications">
        <span class="mail-status "></span>
        <svg class="octicon octicon-bell" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 12v1H0v-1l.73-.58c.77-.77.81-2.55 1.19-4.42C2.69 3.23 6 2 6 2c0-.55.45-1 1-1s1 .45 1 1c0 0 3.39 1.23 4.16 5 .38 1.88.42 3.66 1.19 4.42l.66.58H14zm-7 4c1.11 0 2-.89 2-2H5c0 1.11.89 2 2 2z"/></svg>
</a>
    </div>


    <div class="Header-item position-relative d-none d-lg-flex">
      <details class="details-overlay details-reset">
  <summary class="Header-link"
      aria-label="Create new…"
      data-ga-click="Header, create new, icon:add">
    <svg class="octicon octicon-plus" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 9H7v5H5V9H0V7h5V2h2v5h5v2z"/></svg> <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw">
    
<a role="menuitem" class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a role="menuitem" class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a role="menuitem" class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a role="menuitem" class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>


  <div role="none" class="dropdown-divider"></div>
  <div class="dropdown-header">
    <span title="kezhengzhu/ad-saftgamma">This repository</span>
  </div>
    <a role="menuitem" class="dropdown-item" href="/kezhengzhu/ad-saftgamma/issues/new" data-ga-click="Header, create new issue" data-skip-pjax>
      New issue
    </a>


  </details-menu>
</details>

    </div>

    <div class="Header-item position-relative mr-0 d-none d-lg-flex">
      
<details class="details-overlay details-reset">
  <summary class="Header-link"
    aria-label="View profile and more"
    data-ga-click="Header, show menu, icon:avatar">
    <img alt="@UsmaanHafiz" class="avatar" src="https://avatars1.githubusercontent.com/u/56049634?s=40&amp;v=4" height="20" width="20">
      <include-fragment hidden src="/users/UsmaanHafiz/feature_preview/indicator_check"></include-fragment>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw mt-2" style="width: 180px">
    <div class="header-nav-current-user css-truncate"><a role="menuitem" class="no-underline user-profile-link px-3 pt-2 pb-2 mb-n2 mt-n1 d-block" href="/UsmaanHafiz" data-ga-click="Header, go to profile, text:Signed in as">Signed in as <strong class="css-truncate-target">UsmaanHafiz</strong></a></div>
    <div role="none" class="dropdown-divider"></div>

      <div class="pl-3 pr-3 f6 user-status-container js-user-status-context pb-1" data-url="/users/status?compact=1&amp;link_mentions=0&amp;truncate=1">
        
<div class="js-user-status-container
    user-status-compact rounded-1 px-2 py-1 mt-2
    border
  " data-team-hovercards-enabled>
  <details class="js-user-status-details details-reset details-overlay details-overlay-dark">
    <summary class="btn-link btn-block link-gray no-underline js-toggle-user-status-edit toggle-user-status-edit "
      role="menuitem" data-hydro-click="{&quot;event_type&quot;:&quot;user_profile.click&quot;,&quot;payload&quot;:{&quot;profile_user_id&quot;:44840976,&quot;target&quot;:&quot;EDIT_USER_STATUS&quot;,&quot;user_id&quot;:56049634,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;}}" data-hydro-click-hmac="136b1aab1796b4bfc854794575e01c36ed930ded09010e37c17d269a930cf908">
      <div class="d-flex">
        <div class="f6 lh-condensed user-status-header
          d-inline-block v-align-middle
            user-status-emoji-only-header circle
            pr-2
"
            style="max-width: 29px"
          >
          <div class="user-status-emoji-container flex-shrink-0 mr-1 mt-1 lh-condensed-ultra v-align-bottom" style="">
            <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm4.81 12.81a6.72 6.72 0 01-2.17 1.45c-.83.36-1.72.53-2.64.53-.92 0-1.81-.17-2.64-.53-.81-.34-1.55-.83-2.17-1.45a6.773 6.773 0 01-1.45-2.17A6.59 6.59 0 011.21 8c0-.92.17-1.81.53-2.64.34-.81.83-1.55 1.45-2.17.62-.62 1.36-1.11 2.17-1.45A6.59 6.59 0 018 1.21c.92 0 1.81.17 2.64.53.81.34 1.55.83 2.17 1.45.62.62 1.11 1.36 1.45 2.17.36.83.53 1.72.53 2.64 0 .92-.17 1.81-.53 2.64-.34.81-.83 1.55-1.45 2.17zM4 6.8v-.59c0-.66.53-1.19 1.2-1.19h.59c.66 0 1.19.53 1.19 1.19v.59c0 .67-.53 1.2-1.19 1.2H5.2C4.53 8 4 7.47 4 6.8zm5 0v-.59c0-.66.53-1.19 1.2-1.19h.59c.66 0 1.19.53 1.19 1.19v.59c0 .67-.53 1.2-1.19 1.2h-.59C9.53 8 9 7.47 9 6.8zm4 3.2c-.72 1.88-2.91 3-5 3s-4.28-1.13-5-3c-.14-.39.23-1 .66-1h8.59c.41 0 .89.61.75 1z"/></svg>
          </div>
        </div>
        <div class="
          d-inline-block v-align-middle
          
          
           css-truncate css-truncate-target 
           user-status-message-wrapper f6"
           style="line-height: 20px;" >
          <div class="d-inline-block text-gray-dark v-align-text-top text-left">
              <span class="text-gray ml-2">Set status</span>
          </div>
        </div>
      </div>
    </summary>
    <details-dialog class="details-dialog rounded-1 anim-fade-in fast Box Box--overlay" role="dialog" tabindex="-1">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="position-relative flex-auto js-user-status-form" action="/users/status?compact=1&amp;link_mentions=0&amp;truncate=1" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="YnedvOGcSVmtebNaE6B5bSJkkUaVhOcVYO0S0g0xcpTknJ0dyfgK5flymBTN3JxCTr4w66i+MwUEkiEAV/izDA==" />
        <div class="Box-header bg-gray border-bottom p-3">
          <button class="Box-btn-octicon js-toggle-user-status-edit btn-octicon float-right" type="reset" aria-label="Close dialog" data-close-dialog>
            <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
          </button>
          <h3 class="Box-title f5 text-bold text-gray-dark">Edit status</h3>
        </div>
        <input type="hidden" name="emoji" class="js-user-status-emoji-field" value="">
        <input type="hidden" name="organization_id" class="js-user-status-org-id-field" value="">
        <div class="px-3 py-2 text-gray-dark">
          <div class="js-characters-remaining-container position-relative mt-2">
            <div class="input-group d-table form-group my-0 js-user-status-form-group">
              <span class="input-group-button d-table-cell v-align-middle" style="width: 1%">
                <button type="button" aria-label="Choose an emoji" class="btn-outline btn js-toggle-user-status-emoji-picker btn-open-emoji-picker p-0">
                  <span class="js-user-status-original-emoji" hidden></span>
                  <span class="js-user-status-custom-emoji"></span>
                  <span class="js-user-status-no-emoji-icon" >
                    <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8zm4.81 12.81a6.72 6.72 0 01-2.17 1.45c-.83.36-1.72.53-2.64.53-.92 0-1.81-.17-2.64-.53-.81-.34-1.55-.83-2.17-1.45a6.773 6.773 0 01-1.45-2.17A6.59 6.59 0 011.21 8c0-.92.17-1.81.53-2.64.34-.81.83-1.55 1.45-2.17.62-.62 1.36-1.11 2.17-1.45A6.59 6.59 0 018 1.21c.92 0 1.81.17 2.64.53.81.34 1.55.83 2.17 1.45.62.62 1.11 1.36 1.45 2.17.36.83.53 1.72.53 2.64 0 .92-.17 1.81-.53 2.64-.34.81-.83 1.55-1.45 2.17zM4 6.8v-.59c0-.66.53-1.19 1.2-1.19h.59c.66 0 1.19.53 1.19 1.19v.59c0 .67-.53 1.2-1.19 1.2H5.2C4.53 8 4 7.47 4 6.8zm5 0v-.59c0-.66.53-1.19 1.2-1.19h.59c.66 0 1.19.53 1.19 1.19v.59c0 .67-.53 1.2-1.19 1.2h-.59C9.53 8 9 7.47 9 6.8zm4 3.2c-.72 1.88-2.91 3-5 3s-4.28-1.13-5-3c-.14-.39.23-1 .66-1h8.59c.41 0 .89.61.75 1z"/></svg>
                  </span>
                </button>
              </span>
              <text-expander keys=": @" data-mention-url="/autocomplete/user-suggestions" data-emoji-url="/autocomplete/emoji">
                <input
                  type="text"
                  autocomplete="off"
                  data-no-org-url="/autocomplete/user-suggestions"
                  data-org-url="/suggestions?mention_suggester=1"
                  data-maxlength="80"
                  class="d-table-cell width-full form-control js-user-status-message-field js-characters-remaining-field"
                  placeholder="What's happening?"
                  name="message"
                  value=""
                  aria-label="What is your current status?">
              </text-expander>
              <div class="error">Could not update your status, please try again.</div>
            </div>
            <div style="margin-left: 53px" class="my-1 text-small label-characters-remaining js-characters-remaining" data-suffix="remaining" hidden>
              80 remaining
            </div>
          </div>
          <include-fragment class="js-user-status-emoji-picker" data-url="/users/status/emoji"></include-fragment>
          <div class="overflow-auto ml-n3 mr-n3 px-3 border-bottom" style="max-height: 33vh">
            <div class="user-status-suggestions js-user-status-suggestions collapsed overflow-hidden">
              <h4 class="f6 text-normal my-3">Suggestions:</h4>
              <div class="mx-3 mt-2 clearfix">
                  <div class="float-left col-6">
                      <button type="button" value=":palm_tree:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="palm_tree" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f334.png">🌴</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          On vacation
                        </div>
                      </button>
                      <button type="button" value=":face_with_thermometer:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="face_with_thermometer" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f912.png">🤒</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Out sick
                        </div>
                      </button>
                  </div>
                  <div class="float-left col-6">
                      <button type="button" value=":house:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="house" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3e0.png">🏠</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Working from home
                        </div>
                      </button>
                      <button type="button" value=":dart:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="dart" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3af.png">🎯</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Focusing
                        </div>
                      </button>
                  </div>
              </div>
            </div>
            <div class="user-status-limited-availability-container">
              <div class="form-checkbox my-0">
                <input type="checkbox" name="limited_availability" value="1" class="js-user-status-limited-availability-checkbox" data-default-message="I may be slow to respond." aria-describedby="limited-availability-help-text-truncate-true-compact-true" id="limited-availability-truncate-true-compact-true">
                <label class="d-block f5 text-gray-dark mb-1" for="limited-availability-truncate-true-compact-true">
                  Busy
                </label>
                <p class="note" id="limited-availability-help-text-truncate-true-compact-true">
                  When others mention you, assign you, or request your review,
                  GitHub will let them know that you have limited availability.
                </p>
              </div>
            </div>
          </div>
            

<div class="d-inline-block f5 mr-2 pt-3 pb-2" >
  <div class="d-inline-block mr-1">
    Clear status
  </div>

  <details class="js-user-status-expire-drop-down f6 dropdown details-reset details-overlay d-inline-block mr-2">
    <summary class="f5 btn-link link-gray-dark border px-2 py-1 rounded-1" aria-haspopup="true">
      <div class="js-user-status-expiration-interval-selected d-inline-block v-align-baseline">
        Never
      </div>
      <div class="dropdown-caret"></div>
    </summary>

    <ul class="dropdown-menu dropdown-menu-se pl-0 overflow-auto" style="width: 220px; max-height: 15.5em">
      <li>
        <button type="button" class="btn-link dropdown-item js-user-status-expire-button ws-normal" title="Never">
          <span class="d-inline-block text-bold mb-1">Never</span>
          <div class="f6 lh-condensed">Keep this status until you clear your status or edit your status.</div>
        </button>
      </li>
      <li class="dropdown-divider" role="none"></li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 30 minutes" value="2019-10-27T17:25:11+00:00">
            in 30 minutes
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 1 hour" value="2019-10-27T17:55:11+00:00">
            in 1 hour
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 4 hours" value="2019-10-27T20:55:11+00:00">
            in 4 hours
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="today" value="2019-10-27T23:59:59+00:00">
            today
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="this week" value="2019-10-27T23:59:59+00:00">
            this week
          </button>
        </li>
    </ul>
  </details>
  <input class="js-user-status-expiration-date-input" type="hidden" name="expires_at" value="">
</div>

          <include-fragment class="js-user-status-org-picker" data-url="/users/status/organizations"></include-fragment>
        </div>
        <div class="d-flex flex-items-center flex-justify-between p-3 border-top">
          <button type="submit" disabled class="width-full btn btn-primary mr-2 js-user-status-submit">
            Set status
          </button>
          <button type="button" disabled class="width-full js-clear-user-status-button btn ml-2 ">
            Clear status
          </button>
        </div>
</form>    </details-dialog>
  </details>
</div>

      </div>
      <div role="none" class="dropdown-divider"></div>


    <a role="menuitem" class="dropdown-item" href="/UsmaanHafiz" data-ga-click="Header, go to profile, text:your profile">Your profile</a>


    <a role="menuitem" class="dropdown-item" href="/UsmaanHafiz?tab=repositories" data-ga-click="Header, go to repositories, text:your repositories">Your repositories</a>

    <a role="menuitem" class="dropdown-item" href="/UsmaanHafiz?tab=projects" data-ga-click="Header, go to projects, text:your projects">Your projects</a>

    <a role="menuitem" class="dropdown-item" href="/UsmaanHafiz?tab=stars" data-ga-click="Header, go to starred repos, text:your stars">Your stars</a>
      <a role="menuitem" class="dropdown-item" href="https://gist.github.com/mine" data-ga-click="Header, your gists, text:your gists">Your gists</a>



    <div role="none" class="dropdown-divider"></div>
      
<div id="feature-enrollment-toggle" class="hide-sm hide-md feature-preview-details position-relative">
  <button
    type="button"
    class="dropdown-item btn-link"
    role="menuitem"
    data-feature-preview-trigger-url="/users/UsmaanHafiz/feature_previews"
    data-feature-preview-close-details="{&quot;event_type&quot;:&quot;feature_preview.clicks.close_modal&quot;,&quot;payload&quot;:{&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}"
    data-feature-preview-close-hmac="e84284da56bfcbd2694a6fe137ddc74b236216faf5a92bcabce2a4d1b13a9e1e"
    data-hydro-click="{&quot;event_type&quot;:&quot;feature_preview.clicks.open_modal&quot;,&quot;payload&quot;:{&quot;link_location&quot;:&quot;user_dropdown&quot;,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}"
    data-hydro-click-hmac="50d1611bf403d21232e87b6e1626f14040e470263eeb4a302f9093e58a49e17f"
  >
    Feature preview
  </button>
    <include-fragment hidden src="/users/UsmaanHafiz/feature_preview/indicator_check"></include-fragment>
</div>

    <a role="menuitem" class="dropdown-item" href="https://help.github.com" data-ga-click="Header, go to help, text:help">Help</a>
    <a role="menuitem" class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings">Settings</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="logout-form" action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="/+epc8Cb3DXNedpiaguhHUNCat3IsC4kikSa83oZTwatq+UzqymLFfyE63hEbNIFAYURpa0k7HxyvR9STfAmHw==" />
      
      <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout" role="menuitem">
        Sign out
      </button>
</form>  </details-menu>
</details>

    </div>

  </header>

      

  </div>

  <div id="start-of-content" class="show-on-focus"></div>


    <div id="js-flash-container">

</div>



  <div class="application-main " data-commit-hovercards-enabled>
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main  >
      


  











  <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav pt-0 pt-lg-4 ">
    <div class="repohead-details-container clearfix container-lg p-responsive d-none d-lg-block">

      <ul class="pagehead-actions">




  <li>
    
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-remote="true" class="clearfix js-social-form js-social-container" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="85/4TncRfl6l8Jk+t1VQAPGrrkaYL1jVkDQn/dnja71RLSP95bAWeuLuHCu8kiCuJ7sMrKVDtN2tCPJxQgZfjQ==" />      <input type="hidden" name="repository_id" value="208090094">

      <details class="details-reset details-overlay select-menu float-left">
        <summary class="select-menu-button float-left btn btn-sm btn-with-count" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;WATCH_BUTTON&quot;,&quot;repository_id&quot;:208090094,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}" data-hydro-click-hmac="46f19d4481c4e1a8b5738fc0dd386c0414aa99b2f24f66c3f299368f9a718522" data-ga-click="Repository, click Watch settings, action:blob#show">          <span data-menu-button>
              <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
              Watch
          </span>
</summary>        <details-menu
          class="select-menu-modal position-absolute mt-5"
          style="z-index: 99;">
          <div class="select-menu-header">
            <span class="select-menu-title">Notifications</span>
          </div>
          <div class="select-menu-list">
            <button type="submit" name="do" value="included" class="select-menu-item width-full" aria-checked="true" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Not watching</span>
                <span class="description">Be notified only when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Watch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="release_only" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Releases only</span>
                <span class="description">Be notified of new releases, and when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Unwatch releases
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="subscribed" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Watching</span>
                <span class="description">Be notified of all conversations.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                  Unwatch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="ignore" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Ignoring</span>
                <span class="description">Never be notified.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-mute v-align-text-bottom" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2.81v10.38c0 .67-.81 1-1.28.53L3 10H1c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h2l3.72-3.72C7.19 1.81 8 2.14 8 2.81zm7.53 3.22l-1.06-1.06-1.97 1.97-1.97-1.97-1.06 1.06L11.44 8 9.47 9.97l1.06 1.06 1.97-1.97 1.97 1.97 1.06-1.06L13.56 8l1.97-1.97z"/></svg>
                  Stop ignoring
                </span>
              </div>
            </button>
          </div>
        </details-menu>
      </details>
        <a class="social-count js-social-count"
          href="/kezhengzhu/ad-saftgamma/watchers"
          aria-label="1 user is watching this repository">
          1
        </a>
</form>
  </li>

  <li>
      <div class="js-toggler-container js-social-container starring-container ">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="starred js-social-form" action="/kezhengzhu/ad-saftgamma/unstar" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="YQrxZMoFtKjht0MpvKIsjp2ryfphLg8zGKrErxxeTUdK2OKE65RhBRo1Y/DXKjEPFJJkAcYLfoi08dm3vPcfrg==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count js-toggler-target" aria-label="Unstar this repository" title="Unstar kezhengzhu/ad-saftgamma" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;UNSTAR_BUTTON&quot;,&quot;repository_id&quot;:208090094,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}" data-hydro-click-hmac="8589c6ae2b78087f363525df8545f4a0e4dbef8ed97dcf1cc0bb0f1449dd2be5" data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">        <svg class="octicon octicon-star v-align-text-bottom" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Unstar
</button>        <a class="social-count js-social-count" href="/kezhengzhu/ad-saftgamma/stargazers"
           aria-label="0 users starred this repository">
           0
        </a>
</form>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="unstarred js-social-form" action="/kezhengzhu/ad-saftgamma/star" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="P0ePGiFO2HeUl4MYvc9KQYEdIUIaqfSFo5kdJW+yvSwg5Yh+aJthW5xgHaCmPB+l4fWkp4KOo2CkwXB0vMpgsA==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count js-toggler-target" aria-label="Unstar this repository" title="Star kezhengzhu/ad-saftgamma" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;STAR_BUTTON&quot;,&quot;repository_id&quot;:208090094,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}" data-hydro-click-hmac="a20d9216f72de7bfee64c9547a729b3c90f9481cfe0d1d7212205a118c897445" data-ga-click="Repository, click star button, action:blob#show; text:Star">        <svg class="octicon octicon-star v-align-text-bottom" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Star
</button>        <a class="social-count js-social-count" href="/kezhengzhu/ad-saftgamma/stargazers"
           aria-label="0 users starred this repository">
          0
        </a>
</form>  </div>

  </li>

  <li>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="btn-with-count" action="/kezhengzhu/ad-saftgamma/fork" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="rDoJflg6L69q6OqRN4lerptkWgdoRPPL1+WVSQ2Ga2AiYJ1j/Edd2b4NAFcW+NDnbeGpHBgtGjtWDEW3irA19A==" />
            <button class="btn btn-sm btn-with-count" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;FORK_BUTTON&quot;,&quot;repository_id&quot;:208090094,&quot;client_id&quot;:&quot;2035793917.1562681608&quot;,&quot;originating_request_id&quot;:&quot;CF4D:1AB6C:2F02C94:45EAB13:5DB5CBE7&quot;,&quot;originating_url&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma/blob/master/saftgamma.py&quot;,&quot;referrer&quot;:&quot;https://github.com/kezhengzhu/ad-saftgamma&quot;,&quot;user_id&quot;:56049634}}" data-hydro-click-hmac="d20e3da33f950fb53545b3089fe450293f64cd5ae272087005d6cd0b4afe305f" data-ga-click="Repository, show fork modal, action:blob#show; text:Fork" type="submit" title="Fork your own copy of kezhengzhu/ad-saftgamma to your account" aria-label="Fork your own copy of kezhengzhu/ad-saftgamma to your account">              <svg class="octicon octicon-repo-forked v-align-text-bottom" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 00-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 002 1a1.993 1.993 0 00-1 3.72V6.5l3 3v1.78A1.993 1.993 0 005 15a1.993 1.993 0 001-3.72V9.5l3-3V4.72A1.993 1.993 0 008 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
              Fork
</button></form>
    <a href="/kezhengzhu/ad-saftgamma/network/members" class="social-count"
       aria-label="0 users forked this repository">
      0
    </a>
  </li>
</ul>

      <h1 class="public ">
    <svg class="octicon octicon-repo" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author" itemprop="author"><a class="url fn" rel="author" data-hovercard-type="user" data-hovercard-url="/hovercards?user_id=44840976" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/kezhengzhu">kezhengzhu</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a data-pjax="#js-repo-pjax-container" href="/kezhengzhu/ad-saftgamma">ad-saftgamma</a></strong>
  

</h1>

    </div>
    
<nav class="hx_reponav reponav js-repo-nav js-sidenav-container-pjax container-lg p-responsive d-none d-lg-block"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
    aria-label="Repository"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a class="js-selected-navigation-item selected reponav-item" itemprop="url" data-hotkey="g c" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /kezhengzhu/ad-saftgamma" href="/kezhengzhu/ad-saftgamma">
      <svg class="octicon octicon-code" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" data-hotkey="g i" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /kezhengzhu/ad-saftgamma/issues" href="/kezhengzhu/ad-saftgamma/issues">
        <svg class="octicon octicon-issue-opened" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 011.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg>
        <span itemprop="name">Issues</span>
        <span class="Counter">0</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a data-hotkey="g p" itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /kezhengzhu/ad-saftgamma/pulls" href="/kezhengzhu/ad-saftgamma/pulls">
      <svg class="octicon octicon-git-pull-request" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0010 15a1.993 1.993 0 001-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 00-1 3.72v6.56A1.993 1.993 0 002 15a1.993 1.993 0 001-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">0</span>
      <meta itemprop="position" content="3">
</a>  </span>


    <a data-hotkey="g b" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /kezhengzhu/ad-saftgamma/projects" href="/kezhengzhu/ad-saftgamma/projects">
      <svg class="octicon octicon-project" viewBox="0 0 15 16" version="1.1" width="15" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 00-1 1v14a1 1 0 001 1h13a1 1 0 001-1V1a1 1 0 00-1-1z"/></svg>
      Projects
      <span class="Counter" >0</span>
</a>

    <a class="js-selected-navigation-item reponav-item" data-hotkey="g w" data-selected-links="repo_wiki /kezhengzhu/ad-saftgamma/wiki" href="/kezhengzhu/ad-saftgamma/wiki">
      <svg class="octicon octicon-book" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3 5h4v1H3V5zm0 3h4V7H3v1zm0 2h4V9H3v1zm11-5h-4v1h4V5zm0 2h-4v1h4V7zm0 2h-4v1h4V9zm2-6v9c0 .55-.45 1-1 1H9.5l-1 1-1-1H2c-.55 0-1-.45-1-1V3c0-.55.45-1 1-1h5.5l1 1 1-1H15c.55 0 1 .45 1 1zm-8 .5L7.5 3H2v9h6V3.5zm7-.5H9.5l-.5.5V12h6V3z"/></svg>
      Wiki
</a>
    <a data-skip-pjax="true" class="js-selected-navigation-item reponav-item" data-selected-links="security alerts policy code_scanning /kezhengzhu/ad-saftgamma/security/advisories" href="/kezhengzhu/ad-saftgamma/security/advisories">
      <svg class="octicon octicon-shield" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 2l7-2 7 2v6.02C14 12.69 8.69 16 7 16c-1.69 0-7-3.31-7-7.98V2zm1 .75L7 1l6 1.75v5.268C13 12.104 8.449 15 7 15c-1.449 0-6-2.896-6-6.982V2.75zm1 .75L7 2v12c-1.207 0-5-2.482-5-5.985V3.5z"/></svg>
      Security
</a>
    <a class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph pulse people /kezhengzhu/ad-saftgamma/pulse" href="/kezhengzhu/ad-saftgamma/pulse">
      <svg class="octicon octicon-graph" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
      Insights
</a>

</nav>

  <div class="reponav-wrapper reponav-small d-lg-none">
  <nav class="reponav js-reponav text-center no-wrap"
       itemscope
       itemtype="http://schema.org/BreadcrumbList">

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a class="js-selected-navigation-item selected reponav-item" itemprop="url" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /kezhengzhu/ad-saftgamma" href="/kezhengzhu/ad-saftgamma">
        <span itemprop="name">Code</span>
        <meta itemprop="position" content="1">
</a>    </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /kezhengzhu/ad-saftgamma/issues" href="/kezhengzhu/ad-saftgamma/issues">
          <span itemprop="name">Issues</span>
          <span class="Counter">0</span>
          <meta itemprop="position" content="2">
</a>      </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /kezhengzhu/ad-saftgamma/pulls" href="/kezhengzhu/ad-saftgamma/pulls">
        <span itemprop="name">Pull requests</span>
        <span class="Counter">0</span>
        <meta itemprop="position" content="3">
</a>    </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /kezhengzhu/ad-saftgamma/projects" href="/kezhengzhu/ad-saftgamma/projects">
          <span itemprop="name">Projects</span>
          <span class="Counter">0</span>
          <meta itemprop="position" content="4">
</a>      </span>

      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_wiki /kezhengzhu/ad-saftgamma/wiki" href="/kezhengzhu/ad-saftgamma/wiki">
          <span itemprop="name">Wiki</span>
          <meta itemprop="position" content="5">
</a>      </span>

      <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="security alerts policy code_scanning /kezhengzhu/ad-saftgamma/security/advisories" href="/kezhengzhu/ad-saftgamma/security/advisories">
        <span itemprop="name">Security</span>
        <meta itemprop="position" content="6">
</a>
      <a class="js-selected-navigation-item reponav-item" data-selected-links="pulse /kezhengzhu/ad-saftgamma/pulse" href="/kezhengzhu/ad-saftgamma/pulse">
        Pulse
</a>
      <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
        <a itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="community /kezhengzhu/ad-saftgamma/community" href="/kezhengzhu/ad-saftgamma/community">
          Community
</a>      </span>

  </nav>
</div>


  </div>
<div class="container-lg clearfix new-discussion-timeline experiment-repo-nav  p-responsive">
  <div class="repository-content ">

    
    


  


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/kezhengzhu/ad-saftgamma/blob/cbf0700150217ff388638e3560711e052dca7c31/saftgamma.py">Permalink</a>

    <!-- blob contrib key: blob_contributors:v21:1545e446bcecfd7a9524b149e0511dba -->
      

    <div class="d-flex flex-items-start flex-shrink-0 pb-3 flex-column flex-md-row">
      <span class="d-flex flex-justify-between width-full width-md-auto">
        
<details class="details-reset details-overlay select-menu branch-select-menu  hx_rsm" id="branch-select-menu">
  <summary class="btn btn-sm select-menu-button css-truncate"
           data-hotkey="w"
           title="Switch branches or tags">
    <i>Branch:</i>
    <span class="css-truncate-target" data-menu-button>master</span>
  </summary>

  <details-menu class="select-menu-modal hx_rsm-modal position-absolute" style="z-index: 99;" src="/kezhengzhu/ad-saftgamma/ref-list/master/saftgamma.py?source_action=show&amp;source_controller=blob" preload>
    <include-fragment class="select-menu-loading-overlay anim-pulse">
      <svg height="32" class="octicon octicon-octoface" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M14.7 5.34c.13-.32.55-1.59-.13-3.31 0 0-1.05-.33-3.44 1.3-1-.28-2.07-.32-3.13-.32s-2.13.04-3.13.32c-2.39-1.64-3.44-1.3-3.44-1.3-.68 1.72-.26 2.99-.13 3.31C.49 6.21 0 7.33 0 8.69 0 13.84 3.33 15 7.98 15S16 13.84 16 8.69c0-1.36-.49-2.48-1.3-3.35zM8 14.02c-3.3 0-5.98-.15-5.98-3.35 0-.76.38-1.48 1.02-2.07 1.07-.98 2.9-.46 4.96-.46 2.07 0 3.88-.52 4.96.46.65.59 1.02 1.3 1.02 2.07 0 3.19-2.68 3.35-5.98 3.35zM5.49 9.01c-.66 0-1.2.8-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.54-1.78-1.2-1.78zm5.02 0c-.66 0-1.2.79-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.53-1.78-1.2-1.78z"/></svg>
    </include-fragment>
  </details-menu>
</details>

        <div class="BtnGroup flex-shrink-0 d-md-none">
          <a href="/kezhengzhu/ad-saftgamma/find/master"
                class="js-pjax-capture-input btn btn-sm BtnGroup-item"
                data-pjax
                data-hotkey="t">
            Find file
          </a>
          <clipboard-copy value="saftgamma.py" class="btn btn-sm BtnGroup-item">
            Copy path
          </clipboard-copy>
        </div>
      </span>
      <h2 id="blob-path" class="breadcrumb flex-auto min-width-0 text-normal flex-md-self-center ml-md-2 mr-md-3 my-2 my-md-0">
        <span class="js-repo-root text-bold"><span class="js-path-segment"><a data-pjax="true" href="/kezhengzhu/ad-saftgamma"><span>ad-saftgamma</span></a></span></span><span class="separator">/</span><strong class="final-path">saftgamma.py</strong>
      </h2>

      <div class="BtnGroup flex-shrink-0 d-none d-md-inline-block">
        <a href="/kezhengzhu/ad-saftgamma/find/master"
              class="js-pjax-capture-input btn btn-sm BtnGroup-item"
              data-pjax
              data-hotkey="t">
          Find file
        </a>
        <clipboard-copy value="saftgamma.py" class="btn btn-sm BtnGroup-item">
          Copy path
        </clipboard-copy>
      </div>
    </div>



    <include-fragment src="/kezhengzhu/ad-saftgamma/contributors/master/saftgamma.py" class="Box Box--condensed commit-loader">
      <div class="Box-body bg-blue-light f6">
        Fetching contributors&hellip;
      </div>

      <div class="Box-body d-flex flex-items-center" >
          <img alt="" class="loader-loading mr-2" src="https://github.githubassets.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" height="16" />
        <span class="text-red h6 loader-error">Cannot retrieve contributors at this time</span>
      </div>
</include-fragment>




    <div class="Box mt-3 position-relative">
      
<div class="Box-header py-2 d-flex flex-column flex-shrink-0 flex-md-row flex-md-items-center">
  <div class="text-mono f6 flex-auto pr-3 flex-order-2 flex-md-order-1 mt-2 mt-md-0">

      2721 lines (2270 sloc)
      <span class="file-info-divider"></span>
    97 KB
  </div>

  <div class="d-flex py-1 py-md-0 flex-auto flex-order-1 flex-md-order-2 flex-sm-grow-0 flex-justify-between">

    <div class="BtnGroup">
      <a id="raw-url" class="btn btn-sm BtnGroup-item" href="/kezhengzhu/ad-saftgamma/raw/master/saftgamma.py">Raw</a>
        <a class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b" href="/kezhengzhu/ad-saftgamma/blame/master/saftgamma.py">Blame</a>
      <a rel="nofollow" class="btn btn-sm BtnGroup-item" href="/kezhengzhu/ad-saftgamma/commits/master/saftgamma.py">History</a>
    </div>


    <div>
            <a class="btn-octicon tooltipped tooltipped-nw hide-sm"
               href="https://desktop.github.com"
               aria-label="Open this file in GitHub Desktop"
               data-ga-click="Repository, open with desktop, type:windows">
                <svg class="octicon octicon-device-desktop" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M15 2H1c-.55 0-1 .45-1 1v9c0 .55.45 1 1 1h5.34c-.25.61-.86 1.39-2.34 2h8c-1.48-.61-2.09-1.39-2.34-2H15c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm0 9H1V3h14v8z"/></svg>
            </a>

            <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form js-update-url-with-hash" action="/kezhengzhu/ad-saftgamma/edit/master/saftgamma.py" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="tBLNTPkydd2hYTTPCyNzNYi/k6vAX8Vyd3M6X1pdomXrvi+ndwM+aKM3g2exfqOdzICMjGxFqvI9EvMSQNZONw==" />
              <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
                aria-label="Fork this project and edit the file" data-hotkey="e" data-disable-with>
                <svg class="octicon octicon-pencil" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 011.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
              </button>
</form>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form" action="/kezhengzhu/ad-saftgamma/delete/master/saftgamma.py" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="HN+53o0U5CX4v15GgKAehE5A1ZbyA/spEm/HJKo7fRj3M504OPBiLRrRJxDFKfIdFRC+cMFoG9d3I/ZiqfoNGA==" />
            <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
              aria-label="Fork this project and delete the file" data-disable-with>
              <svg class="octicon octicon-trashcan" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
            </button>
</form>    </div>
  </div>
</div>




      

  <div itemprop="text" class="Box-body p-0 blob-wrapper data type-python ">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>!/usr/bin/env python</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> numpy <span class="pl-k">as</span> np </td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> pandas <span class="pl-k">as</span> pd </td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> time</td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> math <span class="pl-k">import</span> pi,tanh,log10</td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> matplotlib.pyplot <span class="pl-k">as</span> plt</td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> sys</td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> scipy.optimize <span class="pl-k">import</span> fsolve, least_squares</td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class="pl-k">from</span> dervarnp <span class="pl-k">import</span> <span class="pl-k">*</span></td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> defconst <span class="pl-k">as</span> cst</td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line"><span class="pl-k">import</span> methods <span class="pl-k">as</span> mt </td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">System</span>(<span class="pl-c1">object</span>):</td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">293</span>, <span class="pl-smi">pressure</span><span class="pl-k">=</span><span class="pl-c1">101325</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">1000</span>, <span class="pl-k">*</span><span class="pl-smi">args</span>):</td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        System units definitions: temperature = K, pressure = Pa, volume = nm^3</span></td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.pressure <span class="pl-k">=</span> pressure</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.moles <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.n_molecules <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> storage variables for calculations</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.molfrac <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">n_den</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">/</span> (<span class="pl-c1">self</span>.volume <span class="pl-k">*</span> <span class="pl-c1">pow</span>(cst.nmtom, <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_molar_volume</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">molar_v</span>):</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> molar_v <span class="pl-k">/</span> (cst.Na <span class="pl-k">*</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">quick_set</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-k">*</span><span class="pl-smi">args</span>):</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> arg <span class="pl-k">in</span> args:</td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg, <span class="pl-c1">tuple</span>) <span class="pl-k">or</span> <span class="pl-c1">isinstance</span>(arg, <span class="pl-c1">list</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Use iterable tuple or list for quick_set<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg[<span class="pl-c1">0</span>], Component), <span class="pl-s"><span class="pl-pds">&quot;</span>First element of iterable must be an instance of Component class<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg[<span class="pl-c1">1</span>], <span class="pl-c1">int</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Second element of iterable must be of int type<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.moles[arg[<span class="pl-c1">0</span>]] <span class="pl-k">=</span> arg[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[arg[<span class="pl-c1">0</span>]] <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.n_molecules <span class="pl-k">=</span> <span class="pl-c1">sum</span>(<span class="pl-c1">self</span>.moles.values())</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">        nmoles <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.molfrac:</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> nmoles</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">add_comp</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comp</span>, <span class="pl-smi">number</span>):</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(comp, Component), <span class="pl-s"><span class="pl-pds">&quot;</span>Component entry must be an instance of Component class<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(number, <span class="pl-c1">int</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Only integer amount of molecules can be added<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">+=</span> number</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">=</span> number</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.n_molecules <span class="pl-k">=</span> <span class="pl-c1">sum</span>(<span class="pl-c1">self</span>.moles.values())</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">        nmoles <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.molfrac:</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> nmoles</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Get derivatives #################</span></td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">dT</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>both<span class="pl-pds">&#39;</span></span>):</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> Var(T)</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">        fn <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_residual()</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>ideal<span class="pl-pds">&#39;</span></span> <span class="pl-k">or</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_ideal()</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">        da <span class="pl-k">=</span> derivative(fn, <span class="pl-c1">self</span>.temperature)</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> da</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">dV</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>both<span class="pl-pds">&#39;</span></span>):</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(V)</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">        fn <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_residual()</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>ideal<span class="pl-pds">&#39;</span></span> <span class="pl-k">or</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_ideal()</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">        da <span class="pl-k">=</span> derivative(fn, <span class="pl-c1">self</span>.volume)</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">        da <span class="pl-k">=</span> da <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> V</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> da</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">dT2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>both<span class="pl-pds">&#39;</span></span>, <span class="pl-smi">dT</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> Var(T)</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">        fn <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_residual()</td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>ideal<span class="pl-pds">&#39;</span></span> <span class="pl-k">or</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_ideal()</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">        da, d2a <span class="pl-k">=</span> derivative(fn, <span class="pl-c1">self</span>.temperature, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> (da, d2a) <span class="pl-k">if</span> dT <span class="pl-k">else</span> d2a</td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">dV2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>both<span class="pl-pds">&#39;</span></span>, <span class="pl-smi">dV</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(V)</td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">        fn <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_residual()</td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>ideal<span class="pl-pds">&#39;</span></span> <span class="pl-k">or</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_ideal()</td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">        da, d2a <span class="pl-k">=</span> derivative(fn, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">        da <span class="pl-k">=</span> da <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">        d2a <span class="pl-k">=</span> d2a <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> V</td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> (da, d2a) <span class="pl-k">if</span> dV <span class="pl-k">else</span> d2a</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">dTdV</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>both<span class="pl-pds">&#39;</span></span>, <span class="pl-smi">first_order</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(V)</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> Var(T)</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">        fn <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_residual()</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>ideal<span class="pl-pds">&#39;</span></span> <span class="pl-k">or</span> a <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz_ideal()</td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">            fn <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">        da, d2a <span class="pl-k">=</span> derivative(fn, <span class="pl-c1">self</span>.temperature, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">        da[<span class="pl-c1">1</span>] <span class="pl-k">=</span> da[<span class="pl-c1">1</span>] <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">        d2a[<span class="pl-c1">1</span>] <span class="pl-k">=</span> d2a[<span class="pl-c1">1</span>] <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">        d2a[<span class="pl-c1">2</span>] <span class="pl-k">=</span> d2a[<span class="pl-c1">2</span>] <span class="pl-k">/</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> V</td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> (da, d2a) <span class="pl-k">if</span> first_order <span class="pl-k">else</span> d2a</td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> End derivatives #################</span></td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">p_v_isotherm</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">volume</span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">gibbs</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Get pressure profile from volume inputs. Volume in m3 per mol</span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(temperature, <span class="pl-c1">float</span>) <span class="pl-k">or</span> <span class="pl-c1">isinstance</span>(temperature, <span class="pl-c1">int</span>):</td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(volume,<span class="pl-c1">float</span>) <span class="pl-k">or</span> <span class="pl-c1">isinstance</span>(volume, np.ndarray), <span class="pl-s"><span class="pl-pds">&quot;</span>Use floats or numpy array for volume<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">        volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> volume <span class="pl-k">/</span> (cst.Na <span class="pl-k">*</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Case single volume value</span></td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(volume, <span class="pl-c1">float</span>):</td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(volume)</td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">            A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">=</span> <span class="pl-k">-</span>derivative(A,<span class="pl-c1">self</span>.volume,<span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> Reset system back to float</span></td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Case numpy array</span></td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">        old_v <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">        vlen <span class="pl-k">=</span> np.size(volume)</td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> np.zeros(vlen)</td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> gibbs: G <span class="pl-k">=</span> np.zeros(vlen)</td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(volume)</td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span>derivative(A,<span class="pl-c1">self</span>.volume) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> gibbs: G <span class="pl-k">=</span> (A.value <span class="pl-k">+</span> P <span class="pl-k">*</span> <span class="pl-c1">self</span>.volume.value<span class="pl-k">*</span><span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>))<span class="pl-k">/</span><span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Reset system back to float</span></td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> old_v</td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> (P, G) <span class="pl-k">if</span> gibbs <span class="pl-k">else</span> P</td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Get direct properties ###########</span></td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_pressure</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates pressure in units Pa</span></td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where Pressure = NkT/V - dA_res/dV</span></td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">        dArdV <span class="pl-k">=</span> <span class="pl-c1">self</span>.dV(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> (<span class="pl-c1">self</span>.n_molecules <span class="pl-k">/</span> (volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span>) <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> temperature <span class="pl-k">-</span> dArdV)</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> P</td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_entropy</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates entropy in units J/mol K</span></td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where entropy = -dA/dT</span></td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">        dAdT <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT()</td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">        s <span class="pl-k">=</span> <span class="pl-k">-</span>dAdT <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> s</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_u</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates internal energy in units J/mol</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where u = A + TS</span></td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">        dAdT <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT()</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line">        s <span class="pl-k">=</span> <span class="pl-k">-</span>dAdT</td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line">        u <span class="pl-k">=</span> (A <span class="pl-k">+</span> temperature <span class="pl-k">*</span> s) <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> u</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_enthalpy</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates enthalpy in units J/mol</span></td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where H = A + TS + PV</span></td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">        dAdT <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT()</td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">        dAdV <span class="pl-k">=</span> <span class="pl-c1">self</span>.dV()</td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">        s <span class="pl-k">=</span> <span class="pl-k">-</span>dAdT</td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span>dAdV</td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">        h <span class="pl-k">=</span> (A <span class="pl-k">+</span> temperature <span class="pl-k">*</span> s <span class="pl-k">+</span> P <span class="pl-k">*</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span>) <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> h</td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_gibbs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates gibbs free energy in units J/mol</span></td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where gibbs = A + PV</span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line">        dAdV <span class="pl-k">=</span> <span class="pl-c1">self</span>.dV()</td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span>dAdV</td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line">        g <span class="pl-k">=</span> (A <span class="pl-k">+</span> P <span class="pl-k">*</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span>) <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> g</td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_cv</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates isochoric heat capacity in units J/(mol K)</span></td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where cv = (dU/dT)_V</span></td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line">        d2AdT2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT2()</td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line">        cv <span class="pl-k">=</span> <span class="pl-k">-</span>temperature <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> d2AdT2</td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> cv</td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_kappa</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates isothermal compressibility in units m3 / J</span></td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where kappa = -1/V (dV/dP)_T </span></td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line">        d2adv2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.dV2(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line">        v <span class="pl-k">=</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line">        dpdv <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> temperature <span class="pl-k">/</span> <span class="pl-c1">pow</span>(v, <span class="pl-c1">2</span>) <span class="pl-k">-</span> d2adv2</td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line">        kappa <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">1</span> <span class="pl-k">/</span> (v <span class="pl-k">*</span> dpdv)</td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> kappa</td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_cp</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates isobaric heat capacity in units J/(mol K)</span></td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where cp = (dH/dT)_V</span></td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line">        d2r <span class="pl-k">=</span> <span class="pl-c1">self</span>.dTdV(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line">        dt2_id <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT2(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line">        dTdV_r <span class="pl-k">=</span> d2r[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line">        dV2_r <span class="pl-k">=</span> d2r[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line">        dT2 <span class="pl-k">=</span> dt2_id <span class="pl-k">+</span> d2r[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line">        N <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line">        k <span class="pl-k">=</span> cst.k</td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line">        dhdt_v <span class="pl-k">=</span> <span class="pl-k">-</span>T <span class="pl-k">*</span> dT2 <span class="pl-k">+</span> N<span class="pl-k">*</span>k <span class="pl-k">-</span> V <span class="pl-k">*</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line">        dhdv_t <span class="pl-k">=</span> <span class="pl-k">-</span>T <span class="pl-k">*</span> dTdV_r <span class="pl-k">-</span> V <span class="pl-k">*</span> dV2_r</td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line">        dpdt_v <span class="pl-k">=</span> N<span class="pl-k">*</span>k<span class="pl-k">/</span>V <span class="pl-k">-</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line">        dpdv_t <span class="pl-k">=</span> <span class="pl-k">-</span>N<span class="pl-k">*</span>k<span class="pl-k">*</span>T<span class="pl-k">/</span>V<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">-</span> dV2_r</td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line">        cp <span class="pl-k">=</span> dhdt_v <span class="pl-k">-</span> dhdv_t <span class="pl-k">*</span> dpdt_v <span class="pl-k">/</span> dpdv_t</td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line">        cp <span class="pl-k">=</span> cp <span class="pl-k">/</span> N <span class="pl-k">*</span> cst.Na</td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> cp</td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_w</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates speed of sound in units m/s</span></td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where w = sqrt(dP/drho) where rho is in kg/m3</span></td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line">        d2r <span class="pl-k">=</span> <span class="pl-c1">self</span>.dTdV(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line">        dt2_id <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT2(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line">        dTdV_r <span class="pl-k">=</span> d2r[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line">        dV2_r <span class="pl-k">=</span> d2r[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line">        dT2 <span class="pl-k">=</span> dt2_id <span class="pl-k">+</span> d2r[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line">        N <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line">        k <span class="pl-k">=</span> cst.k</td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line">        dsdt_v <span class="pl-k">=</span> <span class="pl-k">-</span>dT2</td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line">        dsdv_t <span class="pl-k">=</span> N<span class="pl-k">*</span>k<span class="pl-k">/</span>V <span class="pl-k">-</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line">        dpdt_v <span class="pl-k">=</span> N<span class="pl-k">*</span>k<span class="pl-k">/</span>V <span class="pl-k">-</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line">        dpdv_t <span class="pl-k">=</span> <span class="pl-k">-</span>N<span class="pl-k">*</span>k<span class="pl-k">*</span>T<span class="pl-k">/</span>V<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">-</span> dV2_r</td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line">        mass <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">            mass <span class="pl-k">+=</span> c.mw <span class="pl-k">*</span> <span class="pl-c1">self</span>.moles[c] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line">        mass <span class="pl-k">=</span> mass <span class="pl-k">*</span> <span class="pl-c1">1e-3</span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line">        dpdv_s <span class="pl-k">=</span> dpdv_t <span class="pl-k">-</span> dpdt_v <span class="pl-k">*</span> dsdv_t <span class="pl-k">/</span> dsdt_v</td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line">        dpdv_s <span class="pl-k">=</span> dpdv_s <span class="pl-k">*</span> N <span class="pl-k">/</span> cst.Na <span class="pl-c"><span class="pl-c">#</span> Pa/(m3/mol)</span></td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line">        vm <span class="pl-k">=</span> V <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> N</td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line">        w <span class="pl-k">=</span> <span class="pl-k">-</span>vm<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">*</span> dpdv_s  <span class="pl-c"><span class="pl-c">#</span> dp/d(rho) J / mol</span></td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line">        w <span class="pl-k">=</span> w <span class="pl-k">/</span> mass <span class="pl-c"><span class="pl-c">#</span> J/kg = m2/s2</span></td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line">        w <span class="pl-k">=</span> sqrt(w)</td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> w</td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_jt</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">molar_volume</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates Joule-Thomson coefficient in units K m3 / J</span></td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        where u_JT = (dT/dP)_h</span></td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> temperature <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(temperature, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line">            temperature <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_volume <span class="pl-k">!=</span> <span class="pl-c1">None</span> <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(molar_volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.set_molar_volume(molar_volume)</td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">elif</span> volume <span class="pl-k">==</span> <span class="pl-c1">None</span> <span class="pl-k">or</span> (<span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(volume, (<span class="pl-c1">int</span>, <span class="pl-c1">float</span>))):</td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line">            volume <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume</td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> volume</td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line">        d2r <span class="pl-k">=</span> <span class="pl-c1">self</span>.dTdV(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line">        dt2_id <span class="pl-k">=</span> <span class="pl-c1">self</span>.dT2(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>id<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line">        dTdV_r <span class="pl-k">=</span> d2r[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">        dV2_r <span class="pl-k">=</span> d2r[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line">        dT2 <span class="pl-k">=</span> dt2_id <span class="pl-k">+</span> d2r[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line">        V <span class="pl-k">=</span> volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line">        N <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line">        k <span class="pl-k">=</span> cst.k</td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line">        dhdt_v <span class="pl-k">=</span> <span class="pl-k">-</span>T <span class="pl-k">*</span> dT2 <span class="pl-k">+</span> N<span class="pl-k">*</span>k <span class="pl-k">-</span> V <span class="pl-k">*</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line">        dhdv_t <span class="pl-k">=</span> <span class="pl-k">-</span>T <span class="pl-k">*</span> dTdV_r <span class="pl-k">-</span> V <span class="pl-k">*</span> dV2_r</td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line">        dpdt_v <span class="pl-k">=</span> N<span class="pl-k">*</span>k<span class="pl-k">/</span>V <span class="pl-k">-</span> dTdV_r</td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line">        dpdv_t <span class="pl-k">=</span> <span class="pl-k">-</span>N<span class="pl-k">*</span>k<span class="pl-k">*</span>T<span class="pl-k">/</span>V<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">-</span> dV2_r</td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line">        dtdp_h <span class="pl-k">=</span> (dpdt_v <span class="pl-k">-</span> dpdv_t <span class="pl-k">*</span> dhdt_v <span class="pl-k">/</span> dhdv_t) <span class="pl-k">**</span> <span class="pl-k">-</span><span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line">        ujt <span class="pl-k">=</span> dtdp_h</td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> ujt</td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> End direct properties ###########</span></td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Calculate critical point ########</span></td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">critical_point</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">initial_t</span><span class="pl-k">=</span><span class="pl-c1">300</span>, <span class="pl-smi">v_nd</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">get_volume</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">get_density</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">print_results</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-smi">solver</span><span class="pl-k">=</span>least_squares, <span class="pl-smi">solver_kwargs</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>bounds<span class="pl-pds">&#39;</span></span>: (<span class="pl-c1">0.1</span>,<span class="pl-c1">10</span>)}, <span class="pl-smi">xtol</span><span class="pl-k">=</span><span class="pl-c1">1e-8</span>, <span class="pl-smi">print_progress</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line">        tscale <span class="pl-k">=</span> initial_t</td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line">        x0 <span class="pl-k">=</span> np.array([<span class="pl-c1">1</span>.])</td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(solver_kwargs, <span class="pl-c1">dict</span>): solver_kwargs <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(v_nd, np.ndarray): v_nd <span class="pl-k">=</span> np.logspace(<span class="pl-k">-</span><span class="pl-c1">4</span>,<span class="pl-k">-</span><span class="pl-c1">2</span>, <span class="pl-c1">50</span>)</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line">        crit_pt <span class="pl-k">=</span> solver(<span class="pl-c1">self</span>.__crit_t, x0, <span class="pl-v">xtol</span><span class="pl-k">=</span>xtol, <span class="pl-v">args</span><span class="pl-k">=</span>(v_nd, tscale, print_progress),<span class="pl-k">**</span>solver_kwargs)</td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_progress: <span class="pl-c1">print</span>()</td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_results: <span class="pl-c1">print</span>(crit_pt)</td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> crit_pt.x[<span class="pl-c1">0</span>]<span class="pl-k">*</span>tscale</td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line">        parr <span class="pl-k">=</span> <span class="pl-c1">self</span>.__crit_t(np.array([T]), v_nd, <span class="pl-c1">1</span>, <span class="pl-c1">False</span>, <span class="pl-c1">True</span>)</td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line">        v_guess <span class="pl-k">=</span> v_nd[parr<span class="pl-k">==</span><span class="pl-c1">max</span>(parr)]</td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line">        crit_v <span class="pl-k">=</span> solver(<span class="pl-c1">self</span>.__crit_v, v_guess, <span class="pl-v">xtol</span><span class="pl-k">=</span>xtol, <span class="pl-v">args</span><span class="pl-k">=</span>(T,), <span class="pl-v">bounds</span><span class="pl-k">=</span>(<span class="pl-c1">min</span>(v_nd), <span class="pl-c1">max</span>(v_nd)))</td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line">        v <span class="pl-k">=</span> crit_v.x[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line">        Var.set_order(<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(mt.m3mol_to_nm(v, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules))</td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span>derivative(A, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> mt.m3mol_to_nm(v, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules)</td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">        results <span class="pl-k">=</span> (P, T)</td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> get_volume: results <span class="pl-k">+=</span> (v,)</td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> get_density: results <span class="pl-k">+=</span> (<span class="pl-c1">1</span><span class="pl-k">/</span>v,)</td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> results</td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__crit_v</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">v</span>, <span class="pl-smi">T</span>):</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line">        v <span class="pl-k">=</span> v[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(mt.m3mol_to_nm(v, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules))</td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line">        _, dpdv <span class="pl-k">=</span> derivative(A, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line">        dpdv <span class="pl-k">=</span> <span class="pl-k">-</span>dpdv <span class="pl-k">/</span> (<span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">6</span>) <span class="pl-k">*</span> cst.Na) <span class="pl-k">/</span> (cst.R <span class="pl-k">*</span> T)</td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> dpdv</td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__crit_t</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">x</span>, <span class="pl-smi">v_nd</span>, <span class="pl-smi">scale</span>, <span class="pl-smi">print_</span>, <span class="pl-smi">getarr</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line">        T <span class="pl-k">=</span> x[<span class="pl-c1">0</span>] <span class="pl-k">*</span> scale</td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> numpy version</span></td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(mt.m3mol_to_nm(v_nd, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules))</td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line">        _,dpdv <span class="pl-k">=</span> derivative(A, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line">        dp <span class="pl-k">=</span> <span class="pl-k">-</span>dpdv <span class="pl-k">/</span> (<span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">6</span>) <span class="pl-k">*</span> cst.Na) <span class="pl-k">/</span> (cst.R <span class="pl-k">*</span> T)</td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_: </td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s">f</span><span class="pl-pds">&#39;</span><span class="pl-s">Current: T = </span><span class="pl-c1">{</span>T<span class="pl-k">:7.3f</span><span class="pl-c1">}</span><span class="pl-s">: max dP/dV = </span><span class="pl-c1">{</span><span class="pl-c1">max</span>(dp)<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-pds">&#39;</span>, <span class="pl-v">end</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-cce">\r</span><span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">max</span>(dp) <span class="pl-k">if</span> getarr <span class="pl-k">==</span> <span class="pl-c1">False</span> <span class="pl-k">else</span> dp</td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Critical point end ##############</span></td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Calculate vapour pressure #######</span></td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__eqpg</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">x</span>, <span class="pl-smi">print_</span>, <span class="pl-smi">scaler</span>):</td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Takes in ndarray to find Pressure1 - Pressure2</span></td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line">        G <span class="pl-k">=</span> []</td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line">        v_ <span class="pl-k">=</span> [x[<span class="pl-c1">0</span>]<span class="pl-k">*</span>scaler[<span class="pl-c1">0</span>], x[<span class="pl-c1">1</span>]<span class="pl-k">*</span>scaler[<span class="pl-c1">1</span>]]</td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i <span class="pl-k">in</span> v_:</td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> mt.m3mol_to_nm(i, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules)</td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line">            A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line">            Pi <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">self</span>.dV()</td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line">            P.append(Pi)</td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line">            G.append((A <span class="pl-k">+</span> Pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.volume<span class="pl-k">*</span><span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>))<span class="pl-k">*</span>cst.Na)</td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_: <span class="pl-c1">print</span>(<span class="pl-s">f</span><span class="pl-pds">&#39;</span><span class="pl-s">Current: v</span><span class="pl-cce">\&#39;</span><span class="pl-s">s = </span><span class="pl-c1">{</span>v_[<span class="pl-c1">0</span>]<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-s">, </span><span class="pl-c1">{</span>v_[<span class="pl-c1">1</span>]<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-s">; dP = </span><span class="pl-c1">{</span>(P[<span class="pl-c1">0</span>]<span class="pl-k">-</span>P[<span class="pl-c1">1</span>])<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-s">, dG = </span><span class="pl-c1">{</span>(G[<span class="pl-c1">0</span>]<span class="pl-k">-</span>G[<span class="pl-c1">1</span>])<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-pds">&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> np.array([(P[<span class="pl-c1">0</span>]<span class="pl-k">-</span>P[<span class="pl-c1">1</span>]), (G[<span class="pl-c1">0</span>]<span class="pl-k">-</span>G[<span class="pl-c1">1</span>])])</td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">vapour_pressure</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">initial_guess</span><span class="pl-k">=</span>(<span class="pl-c1">1e-4</span>,<span class="pl-c1">.1</span>), <span class="pl-smi">get_volume</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">get_gibbs</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">print_results</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-smi">solver</span><span class="pl-k">=</span>least_squares, <span class="pl-smi">solver_kwargs</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>bounds<span class="pl-pds">&#39;</span></span>: ((<span class="pl-c1">0.1</span>,<span class="pl-c1">1e-2</span>),(<span class="pl-c1">1e2</span>,<span class="pl-c1">1e2</span>))}, <span class="pl-smi">print_progress</span><span class="pl-k">=</span><span class="pl-c1">False</span>):</td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(temperature, <span class="pl-c1">float</span>) <span class="pl-k">or</span> <span class="pl-c1">isinstance</span>(temperature, <span class="pl-c1">int</span>):</td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> temperature</td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line">        scaler <span class="pl-k">=</span> initial_guess</td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line">        x0 <span class="pl-k">=</span> np.array([<span class="pl-c1">1</span>.,<span class="pl-c1">1</span>.])</td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(solver_kwargs, <span class="pl-c1">dict</span>): solver_kwargs <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line">        vle <span class="pl-k">=</span> solver(<span class="pl-c1">self</span>.__eqpg, x0, <span class="pl-v">args</span><span class="pl-k">=</span>(print_progress, scaler), <span class="pl-k">**</span>solver_kwargs)</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_progress: <span class="pl-c1">print</span>()</td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> print_results: </td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(<span class="pl-s">f</span><span class="pl-pds">&#39;</span><span class="pl-s">Scaled wrt initial guess of (</span><span class="pl-c1">{</span>scaler[<span class="pl-c1">0</span>]<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-s">,</span><span class="pl-c1">{</span>scaler[<span class="pl-c1">1</span>]<span class="pl-k">:7.3e</span><span class="pl-c1">}</span><span class="pl-s">)</span><span class="pl-pds">&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">print</span>(vle)</td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line">        vlv <span class="pl-k">=</span> (vle.x[<span class="pl-c1">0</span>]<span class="pl-k">*</span>scaler[<span class="pl-c1">0</span>], vle.x[<span class="pl-c1">1</span>]<span class="pl-k">*</span>scaler[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line">        v <span class="pl-k">=</span> vlv[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> Var(mt.m3mol_to_nm(v, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules))</td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.helmholtz()</td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span>derivative(A, <span class="pl-c1">self</span>.volume, <span class="pl-v">order</span><span class="pl-k">=</span><span class="pl-c1">1</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line">        G <span class="pl-k">=</span> (A.value <span class="pl-k">+</span> P <span class="pl-k">*</span> <span class="pl-c1">self</span>.volume.value<span class="pl-k">*</span><span class="pl-c1">pow</span>(cst.nmtom,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> mt.m3mol_to_nm(v, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules)</td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> (P,)</td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> get_volume: result <span class="pl-k">+=</span> (vlv,)</td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> get_gibbs: result <span class="pl-k">+=</span> (G,)</td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Vapour pressure end #############</span></td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Get volume from set P ###########</span></td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__getp</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">x</span>, <span class="pl-smi">targetP</span>):</td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Takes in single value to get P</span></td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line">        x <span class="pl-k">=</span> x[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line">        v <span class="pl-k">=</span> mt.m3mol_to_nm(x, <span class="pl-v">molecules</span><span class="pl-k">=</span><span class="pl-c1">self</span>.n_molecules)</td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.volume <span class="pl-k">=</span> v</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line">        P <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">self</span>.dV()</td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> P <span class="pl-k">-</span> targetP</td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">single_phase_v</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">P</span>, <span class="pl-smi">T</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">print_results</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-smi">get_density</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">use_jac</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">vle_ig</span><span class="pl-k">=</span>(<span class="pl-c1">1e-4</span>, <span class="pl-c1">1e-2</span>), <span class="pl-smi">v_crit</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">v_init</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">supercritical</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-smi">solver_kwargs</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&#39;</span>bounds<span class="pl-pds">&#39;</span></span>: (<span class="pl-c1">1e-4</span>,<span class="pl-c1">1e-1</span>)}):</td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(T, (<span class="pl-c1">int</span>,<span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.temperature <span class="pl-k">=</span> T</td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(P, (<span class="pl-c1">int</span>,<span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> supercritical:</td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line">                mt.checkerr(v_crit <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>, <span class="pl-s"><span class="pl-pds">&quot;</span>For supercritical, please provide v_crit<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line">                v_init <span class="pl-k">=</span> v_crit <span class="pl-k">if</span> v_init <span class="pl-k">is</span> <span class="pl-c1">None</span> <span class="pl-k">else</span> v_init</td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line">                solver_kwargs[<span class="pl-s"><span class="pl-pds">&#39;</span>bounds<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">0.2</span> <span class="pl-k">*</span> v_crit, <span class="pl-c1">1000</span><span class="pl-k">*</span>v_crit)</td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> v_crit <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line">                    vle_ig <span class="pl-k">=</span> (<span class="pl-c1">0.25</span> <span class="pl-k">*</span> v_crit, <span class="pl-c1">100</span> <span class="pl-k">*</span> v_crit)</td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line">                    solver_kwargs[<span class="pl-s"><span class="pl-pds">&#39;</span>bounds<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span> (<span class="pl-c1">0.2</span> <span class="pl-k">*</span> v_crit, <span class="pl-c1">1000</span><span class="pl-k">*</span>v_crit)</td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line">                Pv, vle <span class="pl-k">=</span> <span class="pl-c1">self</span>.vapour_pressure(<span class="pl-v">temperature</span><span class="pl-k">=</span>T, <span class="pl-v">initial_guess</span><span class="pl-k">=</span>vle_ig, <span class="pl-v">get_volume</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line">                v_init <span class="pl-k">=</span> vle[<span class="pl-c1">0</span>] <span class="pl-k">if</span> P <span class="pl-k">&gt;</span> Pv <span class="pl-k">else</span> vle[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line">            getv <span class="pl-k">=</span> least_squares(<span class="pl-c1">self</span>.__getp, v_init, <span class="pl-v">kwargs</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&quot;</span>targetP<span class="pl-pds">&quot;</span></span>: P}, <span class="pl-k">**</span>solver_kwargs)</td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> print_results: <span class="pl-c1">print</span>(getv)</td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> getv.x[<span class="pl-c1">0</span>] <span class="pl-k">if</span> get_density <span class="pl-k">==</span> <span class="pl-c1">False</span> <span class="pl-k">else</span> (getv.x[<span class="pl-c1">0</span>], <span class="pl-c1">1</span><span class="pl-k">/</span>getv.x[<span class="pl-c1">0</span>])</td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(P, np.ndarray), <span class="pl-s"><span class="pl-pds">&quot;</span>Use float/int or ndarray for pressure inputs<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line">        varr <span class="pl-k">=</span> np.zeros(np.size(P))</td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-c1">len</span>(P)):</td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line">            Pv, vle <span class="pl-k">=</span> <span class="pl-c1">self</span>.vapour_pressure(<span class="pl-v">temperature</span><span class="pl-k">=</span>T, <span class="pl-v">initial_guess</span><span class="pl-k">=</span>vle_ig, <span class="pl-v">get_volume</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line">            v_init <span class="pl-k">=</span> vle[<span class="pl-c1">0</span>] <span class="pl-k">if</span> P[i] <span class="pl-k">&gt;</span> Pv <span class="pl-k">else</span> vle[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line">            getv <span class="pl-k">=</span> least_squares(<span class="pl-c1">self</span>.__getp, v_init, <span class="pl-v">kwargs</span><span class="pl-k">=</span>{<span class="pl-s"><span class="pl-pds">&quot;</span>targetP<span class="pl-pds">&quot;</span></span>: P[i]}, <span class="pl-k">**</span>solver_kwargs)</td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line">            varr[i] <span class="pl-k">=</span> getv.x[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> varr <span class="pl-k">if</span> get_density <span class="pl-k">==</span> <span class="pl-c1">False</span> <span class="pl-k">else</span> (varr, <span class="pl-c1">1</span>.<span class="pl-k">/</span>varr)</td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> End v finding ###################</span></td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Function placeholder: this has to be altered for different versions of </span></td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        equation-of-state and systems.</span></td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">42</span></td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Function placeholder: this has to be altered for different versions of </span></td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        equation-of-state and systems.</span></td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">42</span></td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L637" class="blob-num js-line-number" data-line-number="637"></td>
        <td id="LC637" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_residual</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L638" class="blob-num js-line-number" data-line-number="638"></td>
        <td id="LC638" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L639" class="blob-num js-line-number" data-line-number="639"></td>
        <td id="LC639" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Function placeholder: this has to be altered for different versions of </span></td>
      </tr>
      <tr>
        <td id="L640" class="blob-num js-line-number" data-line-number="640"></td>
        <td id="LC640" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        equation-of-state and systems.</span></td>
      </tr>
      <tr>
        <td id="L641" class="blob-num js-line-number" data-line-number="641"></td>
        <td id="LC641" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L642" class="blob-num js-line-number" data-line-number="642"></td>
        <td id="LC642" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">42</span></td>
      </tr>
      <tr>
        <td id="L643" class="blob-num js-line-number" data-line-number="643"></td>
        <td id="LC643" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L644" class="blob-num js-line-number" data-line-number="644"></td>
        <td id="LC644" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">SAFTVRSystem</span>(<span class="pl-e">System</span>):</td>
      </tr>
      <tr>
        <td id="L645" class="blob-num js-line-number" data-line-number="645"></td>
        <td id="LC645" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L646" class="blob-num js-line-number" data-line-number="646"></td>
        <td id="LC646" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    SAFT-VR Mie System as defined by SAFT-VR Mie Equation-of-state</span></td>
      </tr>
      <tr>
        <td id="L647" class="blob-num js-line-number" data-line-number="647"></td>
        <td id="LC647" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    given in ref: J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786</span></td>
      </tr>
      <tr>
        <td id="L648" class="blob-num js-line-number" data-line-number="648"></td>
        <td id="LC648" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L649" class="blob-num js-line-number" data-line-number="649"></td>
        <td id="LC649" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">293</span>, <span class="pl-smi">pressure</span><span class="pl-k">=</span><span class="pl-c1">101325</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">1000</span>):</td>
      </tr>
      <tr>
        <td id="L650" class="blob-num js-line-number" data-line-number="650"></td>
        <td id="LC650" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">super</span>().<span class="pl-c1">__init__</span>(temperature, pressure, volume)</td>
      </tr>
      <tr>
        <td id="L651" class="blob-num js-line-number" data-line-number="651"></td>
        <td id="LC651" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L652" class="blob-num js-line-number" data-line-number="652"></td>
        <td id="LC652" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> stored values</span></td>
      </tr>
      <tr>
        <td id="L653" class="blob-num js-line-number" data-line-number="653"></td>
        <td id="LC653" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.hsd <span class="pl-k">=</span> np.array([[<span class="pl-c1">None</span>]])</td>
      </tr>
      <tr>
        <td id="L654" class="blob-num js-line-number" data-line-number="654"></td>
        <td id="LC654" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L655" class="blob-num js-line-number" data-line-number="655"></td>
        <td id="LC655" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L656" class="blob-num js-line-number" data-line-number="656"></td>
        <td id="LC656" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.array([[<span class="pl-c1">None</span>]])</td>
      </tr>
      <tr>
        <td id="L657" class="blob-num js-line-number" data-line-number="657"></td>
        <td id="LC657" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L658" class="blob-num js-line-number" data-line-number="658"></td>
        <td id="LC658" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L659" class="blob-num js-line-number" data-line-number="659"></td>
        <td id="LC659" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L660" class="blob-num js-line-number" data-line-number="660"></td>
        <td id="LC660" class="blob-code blob-code-inner js-file-line">        nvr <span class="pl-k">=</span> VRMieComponent.n_total()</td>
      </tr>
      <tr>
        <td id="L661" class="blob-num js-line-number" data-line-number="661"></td>
        <td id="LC661" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.hsd <span class="pl-k">=</span> np.zeros((nvr, nvr), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L662" class="blob-num js-line-number" data-line-number="662"></td>
        <td id="LC662" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.zeros((nvr, nvr), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L663" class="blob-num js-line-number" data-line-number="663"></td>
        <td id="LC663" class="blob-code blob-code-inner js-file-line">        segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L664" class="blob-num js-line-number" data-line-number="664"></td>
        <td id="LC664" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L665" class="blob-num js-line-number" data-line-number="665"></td>
        <td id="LC665" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L666" class="blob-num js-line-number" data-line-number="666"></td>
        <td id="LC666" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L667" class="blob-num js-line-number" data-line-number="667"></td>
        <td id="LC667" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> hard-sphere diam</span></td>
      </tr>
      <tr>
        <td id="L668" class="blob-num js-line-number" data-line-number="668"></td>
        <td id="LC668" class="blob-code blob-code-inner js-file-line">            ci <span class="pl-k">=</span> comp.index</td>
      </tr>
      <tr>
        <td id="L669" class="blob-num js-line-number" data-line-number="669"></td>
        <td id="LC669" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.hsd[ci, ci] <span class="pl-k">=</span> comp.hsd(<span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> <span class="pl-c1">1e-10</span></td>
      </tr>
      <tr>
        <td id="L670" class="blob-num js-line-number" data-line-number="670"></td>
        <td id="LC670" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> comp2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L671" class="blob-num js-line-number" data-line-number="671"></td>
        <td id="LC671" class="blob-code blob-code-inner js-file-line">                ci2 <span class="pl-k">=</span> comp2.index</td>
      </tr>
      <tr>
        <td id="L672" class="blob-num js-line-number" data-line-number="672"></td>
        <td id="LC672" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.hsd[ci, ci2] <span class="pl-k">==</span> <span class="pl-c1">0</span>. <span class="pl-k">and</span> <span class="pl-c1">self</span>.hsd[ci2, ci2] <span class="pl-k">!=</span> <span class="pl-c1">0</span>.:</td>
      </tr>
      <tr>
        <td id="L673" class="blob-num js-line-number" data-line-number="673"></td>
        <td id="LC673" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.hsd[ci, ci2] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.hsd[ci,ci] <span class="pl-k">+</span> <span class="pl-c1">self</span>.hsd[ci2, ci2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L674" class="blob-num js-line-number" data-line-number="674"></td>
        <td id="LC674" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.hsd[ci2, ci] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.hsd[ci,ci] <span class="pl-k">+</span> <span class="pl-c1">self</span>.hsd[ci2, ci2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L675" class="blob-num js-line-number" data-line-number="675"></td>
        <td id="LC675" class="blob-code blob-code-inner js-file-line">                <span class="pl-c"><span class="pl-c">#</span>gcomb</span></td>
      </tr>
      <tr>
        <td id="L676" class="blob-num js-line-number" data-line-number="676"></td>
        <td id="LC676" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.gcomb[ci, ci2] <span class="pl-k">=</span> comp <span class="pl-k">+</span> comp2</td>
      </tr>
      <tr>
        <td id="L677" class="blob-num js-line-number" data-line-number="677"></td>
        <td id="LC677" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> segden</span></td>
      </tr>
      <tr>
        <td id="L678" class="blob-num js-line-number" data-line-number="678"></td>
        <td id="LC678" class="blob-code blob-code-inner js-file-line">            segratio <span class="pl-k">+=</span> <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">*</span> comp.ms</td>
      </tr>
      <tr>
        <td id="L679" class="blob-num js-line-number" data-line-number="679"></td>
        <td id="LC679" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> segratio</td>
      </tr>
      <tr>
        <td id="L680" class="blob-num js-line-number" data-line-number="680"></td>
        <td id="LC680" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> segratio <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L681" class="blob-num js-line-number" data-line-number="681"></td>
        <td id="LC681" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L682" class="blob-num js-line-number" data-line-number="682"></td>
        <td id="LC682" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L683" class="blob-num js-line-number" data-line-number="683"></td>
        <td id="LC683" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_ideal() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_res() <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L684" class="blob-num js-line-number" data-line-number="684"></td>
        <td id="LC684" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L685" class="blob-num js-line-number" data-line-number="685"></td>
        <td id="LC685" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L686" class="blob-num js-line-number" data-line-number="686"></td>
        <td id="LC686" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L687" class="blob-num js-line-number" data-line-number="687"></td>
        <td id="LC687" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_residual</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L688" class="blob-num js-line-number" data-line-number="688"></td>
        <td id="LC688" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L689" class="blob-num js-line-number" data-line-number="689"></td>
        <td id="LC689" class="blob-code blob-code-inner js-file-line">        nvr <span class="pl-k">=</span> VRMieComponent.n_total()</td>
      </tr>
      <tr>
        <td id="L690" class="blob-num js-line-number" data-line-number="690"></td>
        <td id="LC690" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.hsd <span class="pl-k">=</span> np.zeros((nvr, nvr), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L691" class="blob-num js-line-number" data-line-number="691"></td>
        <td id="LC691" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.zeros((nvr, nvr), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L692" class="blob-num js-line-number" data-line-number="692"></td>
        <td id="LC692" class="blob-code blob-code-inner js-file-line">        segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L693" class="blob-num js-line-number" data-line-number="693"></td>
        <td id="LC693" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L694" class="blob-num js-line-number" data-line-number="694"></td>
        <td id="LC694" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L695" class="blob-num js-line-number" data-line-number="695"></td>
        <td id="LC695" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L696" class="blob-num js-line-number" data-line-number="696"></td>
        <td id="LC696" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> hard-sphere diam</span></td>
      </tr>
      <tr>
        <td id="L697" class="blob-num js-line-number" data-line-number="697"></td>
        <td id="LC697" class="blob-code blob-code-inner js-file-line">            ci <span class="pl-k">=</span> comp.index</td>
      </tr>
      <tr>
        <td id="L698" class="blob-num js-line-number" data-line-number="698"></td>
        <td id="LC698" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.hsd[ci, ci] <span class="pl-k">=</span> comp.hsd(<span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> <span class="pl-c1">1e-10</span></td>
      </tr>
      <tr>
        <td id="L699" class="blob-num js-line-number" data-line-number="699"></td>
        <td id="LC699" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> comp2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L700" class="blob-num js-line-number" data-line-number="700"></td>
        <td id="LC700" class="blob-code blob-code-inner js-file-line">                ci2 <span class="pl-k">=</span> comp2.index</td>
      </tr>
      <tr>
        <td id="L701" class="blob-num js-line-number" data-line-number="701"></td>
        <td id="LC701" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.hsd[ci, ci2] <span class="pl-k">==</span> <span class="pl-c1">0</span>. <span class="pl-k">and</span> <span class="pl-c1">self</span>.hsd[ci2, ci2] <span class="pl-k">!=</span> <span class="pl-c1">0</span>.:</td>
      </tr>
      <tr>
        <td id="L702" class="blob-num js-line-number" data-line-number="702"></td>
        <td id="LC702" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.hsd[ci, ci2] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.hsd[ci,ci] <span class="pl-k">+</span> <span class="pl-c1">self</span>.hsd[ci2, ci2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L703" class="blob-num js-line-number" data-line-number="703"></td>
        <td id="LC703" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.hsd[ci2, ci] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.hsd[ci,ci] <span class="pl-k">+</span> <span class="pl-c1">self</span>.hsd[ci2, ci2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L704" class="blob-num js-line-number" data-line-number="704"></td>
        <td id="LC704" class="blob-code blob-code-inner js-file-line">                <span class="pl-c"><span class="pl-c">#</span>gcomb</span></td>
      </tr>
      <tr>
        <td id="L705" class="blob-num js-line-number" data-line-number="705"></td>
        <td id="LC705" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.gcomb[ci, ci2] <span class="pl-k">=</span> comp <span class="pl-k">+</span> comp2</td>
      </tr>
      <tr>
        <td id="L706" class="blob-num js-line-number" data-line-number="706"></td>
        <td id="LC706" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> segden</span></td>
      </tr>
      <tr>
        <td id="L707" class="blob-num js-line-number" data-line-number="707"></td>
        <td id="LC707" class="blob-code blob-code-inner js-file-line">            segratio <span class="pl-k">+=</span> <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">*</span> comp.ms</td>
      </tr>
      <tr>
        <td id="L708" class="blob-num js-line-number" data-line-number="708"></td>
        <td id="LC708" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> segratio</td>
      </tr>
      <tr>
        <td id="L709" class="blob-num js-line-number" data-line-number="709"></td>
        <td id="LC709" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> segratio <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L710" class="blob-num js-line-number" data-line-number="710"></td>
        <td id="LC710" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L711" class="blob-num js-line-number" data-line-number="711"></td>
        <td id="LC711" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L712" class="blob-num js-line-number" data-line-number="712"></td>
        <td id="LC712" class="blob-code blob-code-inner js-file-line">        a <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_res()</td>
      </tr>
      <tr>
        <td id="L713" class="blob-num js-line-number" data-line-number="713"></td>
        <td id="LC713" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> a <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L714" class="blob-num js-line-number" data-line-number="714"></td>
        <td id="LC714" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L715" class="blob-num js-line-number" data-line-number="715"></td>
        <td id="LC715" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L716" class="blob-num js-line-number" data-line-number="716"></td>
        <td id="LC716" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L717" class="blob-num js-line-number" data-line-number="717"></td>
        <td id="LC717" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L718" class="blob-num js-line-number" data-line-number="718"></td>
        <td id="LC718" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L719" class="blob-num js-line-number" data-line-number="719"></td>
        <td id="LC719" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L720" class="blob-num js-line-number" data-line-number="720"></td>
        <td id="LC720" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L721" class="blob-num js-line-number" data-line-number="721"></td>
        <td id="LC721" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L722" class="blob-num js-line-number" data-line-number="722"></td>
        <td id="LC722" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L723" class="blob-num js-line-number" data-line-number="723"></td>
        <td id="LC723" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_ideal()</td>
      </tr>
      <tr>
        <td id="L724" class="blob-num js-line-number" data-line-number="724"></td>
        <td id="LC724" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L725" class="blob-num js-line-number" data-line-number="725"></td>
        <td id="LC725" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L726" class="blob-num js-line-number" data-line-number="726"></td>
        <td id="LC726" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L727" class="blob-num js-line-number" data-line-number="727"></td>
        <td id="LC727" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L728" class="blob-num js-line-number" data-line-number="728"></td>
        <td id="LC728" class="blob-code blob-code-inner js-file-line">        mole_tol <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L729" class="blob-num js-line-number" data-line-number="729"></td>
        <td id="LC729" class="blob-code blob-code-inner js-file-line">        cpideal <span class="pl-k">=</span> <span class="pl-c1">True</span></td>
      </tr>
      <tr>
        <td id="L730" class="blob-num js-line-number" data-line-number="730"></td>
        <td id="LC730" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L731" class="blob-num js-line-number" data-line-number="731"></td>
        <td id="LC731" class="blob-code blob-code-inner js-file-line">            cpint <span class="pl-k">=</span> comp.cp_int(<span class="pl-c1">self</span>.temperature)</td>
      </tr>
      <tr>
        <td id="L732" class="blob-num js-line-number" data-line-number="732"></td>
        <td id="LC732" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> cpint <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L733" class="blob-num js-line-number" data-line-number="733"></td>
        <td id="LC733" class="blob-code blob-code-inner js-file-line">                cpideal <span class="pl-k">=</span> <span class="pl-c1">False</span></td>
      </tr>
      <tr>
        <td id="L734" class="blob-num js-line-number" data-line-number="734"></td>
        <td id="LC734" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L735" class="blob-num js-line-number" data-line-number="735"></td>
        <td id="LC735" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> cpideal <span class="pl-k">==</span> <span class="pl-c1">False</span>:</td>
      </tr>
      <tr>
        <td id="L736" class="blob-num js-line-number" data-line-number="736"></td>
        <td id="LC736" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L737" class="blob-num js-line-number" data-line-number="737"></td>
        <td id="LC737" class="blob-code blob-code-inner js-file-line">                debrogv <span class="pl-k">=</span> <span class="pl-c1">pow</span>(comp.thdebroglie(<span class="pl-c1">self</span>.temperature),<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L738" class="blob-num js-line-number" data-line-number="738"></td>
        <td id="LC738" class="blob-code blob-code-inner js-file-line">                molfrac <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[comp]</td>
      </tr>
      <tr>
        <td id="L739" class="blob-num js-line-number" data-line-number="739"></td>
        <td id="LC739" class="blob-code blob-code-inner js-file-line">                nden <span class="pl-k">=</span> molfrac <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L740" class="blob-num js-line-number" data-line-number="740"></td>
        <td id="LC740" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">=</span> result <span class="pl-k">+</span> molfrac <span class="pl-k">*</span> log(nden <span class="pl-k">*</span> debrogv) </td>
      </tr>
      <tr>
        <td id="L741" class="blob-num js-line-number" data-line-number="741"></td>
        <td id="LC741" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">=</span> result <span class="pl-k">-</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L742" class="blob-num js-line-number" data-line-number="742"></td>
        <td id="LC742" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> result <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L743" class="blob-num js-line-number" data-line-number="743"></td>
        <td id="LC743" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L744" class="blob-num js-line-number" data-line-number="744"></td>
        <td id="LC744" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L745" class="blob-num js-line-number" data-line-number="745"></td>
        <td id="LC745" class="blob-code blob-code-inner js-file-line">            h0 <span class="pl-k">=</span> comp.ref[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L746" class="blob-num js-line-number" data-line-number="746"></td>
        <td id="LC746" class="blob-code blob-code-inner js-file-line">            s0 <span class="pl-k">=</span> comp.ref[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L747" class="blob-num js-line-number" data-line-number="747"></td>
        <td id="LC747" class="blob-code blob-code-inner js-file-line">            tref <span class="pl-k">=</span> comp.ref[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L748" class="blob-num js-line-number" data-line-number="748"></td>
        <td id="LC748" class="blob-code blob-code-inner js-file-line">            pref <span class="pl-k">=</span> comp.ref[<span class="pl-c1">3</span>]</td>
      </tr>
      <tr>
        <td id="L749" class="blob-num js-line-number" data-line-number="749"></td>
        <td id="LC749" class="blob-code blob-code-inner js-file-line">            vref <span class="pl-k">=</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> tref <span class="pl-k">/</span> pref</td>
      </tr>
      <tr>
        <td id="L750" class="blob-num js-line-number" data-line-number="750"></td>
        <td id="LC750" class="blob-code blob-code-inner js-file-line">            t <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L751" class="blob-num js-line-number" data-line-number="751"></td>
        <td id="LC751" class="blob-code blob-code-inner js-file-line">            xi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[comp]</td>
      </tr>
      <tr>
        <td id="L752" class="blob-num js-line-number" data-line-number="752"></td>
        <td id="LC752" class="blob-code blob-code-inner js-file-line">            n_mol <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">/</span> cst.Na</td>
      </tr>
      <tr>
        <td id="L753" class="blob-num js-line-number" data-line-number="753"></td>
        <td id="LC753" class="blob-code blob-code-inner js-file-line">            V <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L754" class="blob-num js-line-number" data-line-number="754"></td>
        <td id="LC754" class="blob-code blob-code-inner js-file-line">            cpint <span class="pl-k">=</span> comp.cp_int(t)</td>
      </tr>
      <tr>
        <td id="L755" class="blob-num js-line-number" data-line-number="755"></td>
        <td id="LC755" class="blob-code blob-code-inner js-file-line">            cptint <span class="pl-k">=</span> comp.cp_t_int(t)</td>
      </tr>
      <tr>
        <td id="L756" class="blob-num js-line-number" data-line-number="756"></td>
        <td id="LC756" class="blob-code blob-code-inner js-file-line">            </td>
      </tr>
      <tr>
        <td id="L757" class="blob-num js-line-number" data-line-number="757"></td>
        <td id="LC757" class="blob-code blob-code-inner js-file-line">            si <span class="pl-k">=</span> cptint <span class="pl-k">-</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> log(xi <span class="pl-k">*</span> t <span class="pl-k">*</span> vref <span class="pl-k">*</span> n_mol <span class="pl-k">/</span> (tref<span class="pl-k">*</span>V)) <span class="pl-k">+</span> s0</td>
      </tr>
      <tr>
        <td id="L758" class="blob-num js-line-number" data-line-number="758"></td>
        <td id="LC758" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L759" class="blob-num js-line-number" data-line-number="759"></td>
        <td id="LC759" class="blob-code blob-code-inner js-file-line">            ai_mol <span class="pl-k">=</span> cpint <span class="pl-k">-</span> t <span class="pl-k">*</span> si <span class="pl-k">-</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> t <span class="pl-k">+</span> h0 <span class="pl-c"><span class="pl-c">#</span> per mol</span></td>
      </tr>
      <tr>
        <td id="L760" class="blob-num js-line-number" data-line-number="760"></td>
        <td id="LC760" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">=</span> result <span class="pl-k">+</span> xi <span class="pl-k">*</span> n_mol <span class="pl-k">*</span> ai_mol</td>
      </tr>
      <tr>
        <td id="L761" class="blob-num js-line-number" data-line-number="761"></td>
        <td id="LC761" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L762" class="blob-num js-line-number" data-line-number="762"></td>
        <td id="LC762" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L763" class="blob-num js-line-number" data-line-number="763"></td>
        <td id="LC763" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L764" class="blob-num js-line-number" data-line-number="764"></td>
        <td id="LC764" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_res</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L765" class="blob-num js-line-number" data-line-number="765"></td>
        <td id="LC765" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.a_mono() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_chain() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_assoc()</td>
      </tr>
      <tr>
        <td id="L766" class="blob-num js-line-number" data-line-number="766"></td>
        <td id="LC766" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L767" class="blob-num js-line-number" data-line-number="767"></td>
        <td id="LC767" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_mono</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L768" class="blob-num js-line-number" data-line-number="768"></td>
        <td id="LC768" class="blob-code blob-code-inner js-file-line">        a_m <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_hs() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_1() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_2() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_3()</td>
      </tr>
      <tr>
        <td id="L769" class="blob-num js-line-number" data-line-number="769"></td>
        <td id="LC769" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> a_m</td>
      </tr>
      <tr>
        <td id="L770" class="blob-num js-line-number" data-line-number="770"></td>
        <td id="LC770" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L771" class="blob-num js-line-number" data-line-number="771"></td>
        <td id="LC771" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_chain</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L772" class="blob-num js-line-number" data-line-number="772"></td>
        <td id="LC772" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L773" class="blob-num js-line-number" data-line-number="773"></td>
        <td id="LC773" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L774" class="blob-num js-line-number" data-line-number="774"></td>
        <td id="LC774" class="blob-code blob-code-inner js-file-line">            molfrac <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c]</td>
      </tr>
      <tr>
        <td id="L775" class="blob-num js-line-number" data-line-number="775"></td>
        <td id="LC775" class="blob-code blob-code-inner js-file-line">            ms <span class="pl-k">=</span> c.ms</td>
      </tr>
      <tr>
        <td id="L776" class="blob-num js-line-number" data-line-number="776"></td>
        <td id="LC776" class="blob-code blob-code-inner js-file-line">            gmieii <span class="pl-k">=</span> <span class="pl-c1">self</span>.__gmieij(c.index, c.index)</td>
      </tr>
      <tr>
        <td id="L777" class="blob-num js-line-number" data-line-number="777"></td>
        <td id="LC777" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">+=</span> molfrac <span class="pl-k">*</span> (ms <span class="pl-k">-</span> <span class="pl-c1">1</span>) <span class="pl-k">*</span> log(gmieii)</td>
      </tr>
      <tr>
        <td id="L778" class="blob-num js-line-number" data-line-number="778"></td>
        <td id="LC778" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-k">-</span>result</td>
      </tr>
      <tr>
        <td id="L779" class="blob-num js-line-number" data-line-number="779"></td>
        <td id="LC779" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L780" class="blob-num js-line-number" data-line-number="780"></td>
        <td id="LC780" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L781" class="blob-num js-line-number" data-line-number="781"></td>
        <td id="LC781" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_assoc</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L782" class="blob-num js-line-number" data-line-number="782"></td>
        <td id="LC782" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">0</span> <span class="pl-c"><span class="pl-c">#</span> placeholder self.volume ** 3 * 0.1 + exp(-self.volume)</span></td>
      </tr>
      <tr>
        <td id="L783" class="blob-num js-line-number" data-line-number="783"></td>
        <td id="LC783" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L784" class="blob-num js-line-number" data-line-number="784"></td>
        <td id="LC784" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for all</span></td>
      </tr>
      <tr>
        <td id="L785" class="blob-num js-line-number" data-line-number="785"></td>
        <td id="LC785" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_m</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">m</span>):</td>
      </tr>
      <tr>
        <td id="L786" class="blob-num js-line-number" data-line-number="786"></td>
        <td id="LC786" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L787" class="blob-num js-line-number" data-line-number="787"></td>
        <td id="LC787" class="blob-code blob-code-inner js-file-line">        gsum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L788" class="blob-num js-line-number" data-line-number="788"></td>
        <td id="LC788" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L789" class="blob-num js-line-number" data-line-number="789"></td>
        <td id="LC789" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c] <span class="pl-k">*</span> c.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L790" class="blob-num js-line-number" data-line-number="790"></td>
        <td id="LC790" class="blob-code blob-code-inner js-file-line">            gsum <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(<span class="pl-c1">self</span>.hsd[c.index, c.index], m)</td>
      </tr>
      <tr>
        <td id="L791" class="blob-num js-line-number" data-line-number="791"></td>
        <td id="LC791" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> gsum</td>
      </tr>
      <tr>
        <td id="L792" class="blob-num js-line-number" data-line-number="792"></td>
        <td id="LC792" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L793" class="blob-num js-line-number" data-line-number="793"></td>
        <td id="LC793" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_x</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L794" class="blob-num js-line-number" data-line-number="794"></td>
        <td id="LC794" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L795" class="blob-num js-line-number" data-line-number="795"></td>
        <td id="LC795" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L796" class="blob-num js-line-number" data-line-number="796"></td>
        <td id="LC796" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L797" class="blob-num js-line-number" data-line-number="797"></td>
        <td id="LC797" class="blob-code blob-code-inner js-file-line">            c1i <span class="pl-k">=</span> c1.index</td>
      </tr>
      <tr>
        <td id="L798" class="blob-num js-line-number" data-line-number="798"></td>
        <td id="LC798" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L799" class="blob-num js-line-number" data-line-number="799"></td>
        <td id="LC799" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L800" class="blob-num js-line-number" data-line-number="800"></td>
        <td id="LC800" class="blob-code blob-code-inner js-file-line">                c2i <span class="pl-k">=</span> c2.index</td>
      </tr>
      <tr>
        <td id="L801" class="blob-num js-line-number" data-line-number="801"></td>
        <td id="LC801" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L802" class="blob-num js-line-number" data-line-number="802"></td>
        <td id="LC802" class="blob-code blob-code-inner js-file-line">                dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L803" class="blob-num js-line-number" data-line-number="803"></td>
        <td id="LC803" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L804" class="blob-num js-line-number" data-line-number="804"></td>
        <td id="LC804" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L805" class="blob-num js-line-number" data-line-number="805"></td>
        <td id="LC805" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L806" class="blob-num js-line-number" data-line-number="806"></td>
        <td id="LC806" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_sx</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L807" class="blob-num js-line-number" data-line-number="807"></td>
        <td id="LC807" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L808" class="blob-num js-line-number" data-line-number="808"></td>
        <td id="LC808" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L809" class="blob-num js-line-number" data-line-number="809"></td>
        <td id="LC809" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L810" class="blob-num js-line-number" data-line-number="810"></td>
        <td id="LC810" class="blob-code blob-code-inner js-file-line">            c1i <span class="pl-k">=</span> c1.index</td>
      </tr>
      <tr>
        <td id="L811" class="blob-num js-line-number" data-line-number="811"></td>
        <td id="LC811" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L812" class="blob-num js-line-number" data-line-number="812"></td>
        <td id="LC812" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L813" class="blob-num js-line-number" data-line-number="813"></td>
        <td id="LC813" class="blob-code blob-code-inner js-file-line">                c2i <span class="pl-k">=</span> c2.index</td>
      </tr>
      <tr>
        <td id="L814" class="blob-num js-line-number" data-line-number="814"></td>
        <td id="LC814" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L815" class="blob-num js-line-number" data-line-number="815"></td>
        <td id="LC815" class="blob-code blob-code-inner js-file-line">                skl <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i].sigma <span class="pl-k">*</span> cst.atom</td>
      </tr>
      <tr>
        <td id="L816" class="blob-num js-line-number" data-line-number="816"></td>
        <td id="LC816" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> <span class="pl-c1">pow</span>(skl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L817" class="blob-num js-line-number" data-line-number="817"></td>
        <td id="LC817" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L818" class="blob-num js-line-number" data-line-number="818"></td>
        <td id="LC818" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L819" class="blob-num js-line-number" data-line-number="819"></td>
        <td id="LC819" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__bkl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L820" class="blob-num js-line-number" data-line-number="820"></td>
        <td id="LC820" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L821" class="blob-num js-line-number" data-line-number="821"></td>
        <td id="LC821" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Use 2 groups as input</span></td>
      </tr>
      <tr>
        <td id="L822" class="blob-num js-line-number" data-line-number="822"></td>
        <td id="LC822" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        set rep = True for lam_r, rep = False for lam_a</span></td>
      </tr>
      <tr>
        <td id="L823" class="blob-num js-line-number" data-line-number="823"></td>
        <td id="LC823" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        2 * pi * segden * dkl^3 epsikl * bkl_sum(lambda_r or a, xi_x)</span></td>
      </tr>
      <tr>
        <td id="L824" class="blob-num js-line-number" data-line-number="824"></td>
        <td id="LC824" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L825" class="blob-num js-line-number" data-line-number="825"></td>
        <td id="LC825" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L826" class="blob-num js-line-number" data-line-number="826"></td>
        <td id="LC826" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L827" class="blob-num js-line-number" data-line-number="827"></td>
        <td id="LC827" class="blob-code blob-code-inner js-file-line">        dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L828" class="blob-num js-line-number" data-line-number="828"></td>
        <td id="LC828" class="blob-code blob-code-inner js-file-line">        dkl_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L829" class="blob-num js-line-number" data-line-number="829"></td>
        <td id="LC829" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L830" class="blob-num js-line-number" data-line-number="830"></td>
        <td id="LC830" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L831" class="blob-num js-line-number" data-line-number="831"></td>
        <td id="LC831" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(dkl)</td>
      </tr>
      <tr>
        <td id="L832" class="blob-num js-line-number" data-line-number="832"></td>
        <td id="LC832" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L833" class="blob-num js-line-number" data-line-number="833"></td>
        <td id="LC833" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L834" class="blob-num js-line-number" data-line-number="834"></td>
        <td id="LC834" class="blob-code blob-code-inner js-file-line">        bsum <span class="pl-k">=</span> mt.bkl_sum(xix, x0kl, lam)</td>
      </tr>
      <tr>
        <td id="L835" class="blob-num js-line-number" data-line-number="835"></td>
        <td id="LC835" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span>pi<span class="pl-k">*</span>segden<span class="pl-k">*</span>dkl_3<span class="pl-k">*</span>epsikl<span class="pl-k">*</span>bsum</td>
      </tr>
      <tr>
        <td id="L836" class="blob-num js-line-number" data-line-number="836"></td>
        <td id="LC836" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L837" class="blob-num js-line-number" data-line-number="837"></td>
        <td id="LC837" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L838" class="blob-num js-line-number" data-line-number="838"></td>
        <td id="LC838" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L839" class="blob-num js-line-number" data-line-number="839"></td>
        <td id="LC839" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__as1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L840" class="blob-num js-line-number" data-line-number="840"></td>
        <td id="LC840" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L841" class="blob-num js-line-number" data-line-number="841"></td>
        <td id="LC841" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - 2 * pi * segden * (epsikl * dkl^3/ (lam - 3)) * (1 - xi_x_eff/2)/(1-xi_x_eff)^3</span></td>
      </tr>
      <tr>
        <td id="L842" class="blob-num js-line-number" data-line-number="842"></td>
        <td id="LC842" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L843" class="blob-num js-line-number" data-line-number="843"></td>
        <td id="LC843" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L844" class="blob-num js-line-number" data-line-number="844"></td>
        <td id="LC844" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L845" class="blob-num js-line-number" data-line-number="845"></td>
        <td id="LC845" class="blob-code blob-code-inner js-file-line">        dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L846" class="blob-num js-line-number" data-line-number="846"></td>
        <td id="LC846" class="blob-code blob-code-inner js-file-line">        dkl_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L847" class="blob-num js-line-number" data-line-number="847"></td>
        <td id="LC847" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L848" class="blob-num js-line-number" data-line-number="848"></td>
        <td id="LC848" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L849" class="blob-num js-line-number" data-line-number="849"></td>
        <td id="LC849" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L850" class="blob-num js-line-number" data-line-number="850"></td>
        <td id="LC850" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L851" class="blob-num js-line-number" data-line-number="851"></td>
        <td id="LC851" class="blob-code blob-code-inner js-file-line">        xixeff <span class="pl-k">=</span> mt.xi_x_eff(xix, lam)</td>
      </tr>
      <tr>
        <td id="L852" class="blob-num js-line-number" data-line-number="852"></td>
        <td id="LC852" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L853" class="blob-num js-line-number" data-line-number="853"></td>
        <td id="LC853" class="blob-code blob-code-inner js-file-line">        num <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff<span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L854" class="blob-num js-line-number" data-line-number="854"></td>
        <td id="LC854" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L855" class="blob-num js-line-number" data-line-number="855"></td>
        <td id="LC855" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L856" class="blob-num js-line-number" data-line-number="856"></td>
        <td id="LC856" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> segden <span class="pl-k">*</span> (epsikl <span class="pl-k">*</span> dkl_3 <span class="pl-k">/</span> (lam <span class="pl-k">-</span> <span class="pl-c1">3</span>)) <span class="pl-k">*</span> (num<span class="pl-k">/</span>den)</td>
      </tr>
      <tr>
        <td id="L857" class="blob-num js-line-number" data-line-number="857"></td>
        <td id="LC857" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L858" class="blob-num js-line-number" data-line-number="858"></td>
        <td id="LC858" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L859" class="blob-num js-line-number" data-line-number="859"></td>
        <td id="LC859" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L860" class="blob-num js-line-number" data-line-number="860"></td>
        <td id="LC860" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for mono</span></td>
      </tr>
      <tr>
        <td id="L861" class="blob-num js-line-number" data-line-number="861"></td>
        <td id="LC861" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_hs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L862" class="blob-num js-line-number" data-line-number="862"></td>
        <td id="LC862" class="blob-code blob-code-inner js-file-line">        xi1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">1</span>) <span class="pl-c"><span class="pl-c">#</span> nm-2</span></td>
      </tr>
      <tr>
        <td id="L863" class="blob-num js-line-number" data-line-number="863"></td>
        <td id="LC863" class="blob-code blob-code-inner js-file-line">        xi2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">2</span>) <span class="pl-c"><span class="pl-c">#</span> nm-1</span></td>
      </tr>
      <tr>
        <td id="L864" class="blob-num js-line-number" data-line-number="864"></td>
        <td id="LC864" class="blob-code blob-code-inner js-file-line">        xi3 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">3</span>) <span class="pl-c"><span class="pl-c">#</span> dimless</span></td>
      </tr>
      <tr>
        <td id="L865" class="blob-num js-line-number" data-line-number="865"></td>
        <td id="LC865" class="blob-code blob-code-inner js-file-line">        xi0 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">0</span>) <span class="pl-c"><span class="pl-c">#</span> nm-3</span></td>
      </tr>
      <tr>
        <td id="L866" class="blob-num js-line-number" data-line-number="866"></td>
        <td id="LC866" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L867" class="blob-num js-line-number" data-line-number="867"></td>
        <td id="LC867" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">len</span>(<span class="pl-c1">self</span>.moles) <span class="pl-k">==</span> <span class="pl-c1">1</span>:</td>
      </tr>
      <tr>
        <td id="L868" class="blob-num js-line-number" data-line-number="868"></td>
        <td id="LC868" class="blob-code blob-code-inner js-file-line">            t1 <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L869" class="blob-num js-line-number" data-line-number="869"></td>
        <td id="LC869" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L870" class="blob-num js-line-number" data-line-number="870"></td>
        <td id="LC870" class="blob-code blob-code-inner js-file-line">            t1 <span class="pl-k">=</span> (<span class="pl-c1">pow</span>(xi2, <span class="pl-c1">3</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(xi3, <span class="pl-c1">2</span>) <span class="pl-k">-</span> xi0) <span class="pl-k">*</span> log(<span class="pl-c1">1</span><span class="pl-k">-</span>xi3) <span class="pl-c"><span class="pl-c">#</span> t1 is HUGE, and because of numerical discrepancy, even though</span></td>
      </tr>
      <tr>
        <td id="L871" class="blob-num js-line-number" data-line-number="871"></td>
        <td id="LC871" class="blob-code blob-code-inner js-file-line">                                                                <span class="pl-c"><span class="pl-c">#</span> if there is only 1 component, t1 approx 0,</span></td>
      </tr>
      <tr>
        <td id="L872" class="blob-num js-line-number" data-line-number="872"></td>
        <td id="LC872" class="blob-code blob-code-inner js-file-line">                                                                <span class="pl-c"><span class="pl-c">#</span> but because each term is ~ 10^27 so discrepancies could be millions even</span></td>
      </tr>
      <tr>
        <td id="L873" class="blob-num js-line-number" data-line-number="873"></td>
        <td id="LC873" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> xi1 <span class="pl-k">*</span> xi2 <span class="pl-k">/</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> xi3)</td>
      </tr>
      <tr>
        <td id="L874" class="blob-num js-line-number" data-line-number="874"></td>
        <td id="LC874" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(xi2, <span class="pl-c1">3</span>) <span class="pl-k">/</span> (xi3 <span class="pl-k">*</span> <span class="pl-c1">pow</span>((<span class="pl-c1">1</span><span class="pl-k">-</span>xi3),<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L875" class="blob-num js-line-number" data-line-number="875"></td>
        <td id="LC875" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> print(&#39;t2 + t3 gives&#39;, t2+t3)</span></td>
      </tr>
      <tr>
        <td id="L876" class="blob-num js-line-number" data-line-number="876"></td>
        <td id="LC876" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> print(&#39;With pure imp:&#39;,(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))</span></td>
      </tr>
      <tr>
        <td id="L877" class="blob-num js-line-number" data-line-number="877"></td>
        <td id="LC877" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L878" class="blob-num js-line-number" data-line-number="878"></td>
        <td id="LC878" class="blob-code blob-code-inner js-file-line">        xiterm <span class="pl-k">=</span> t1 <span class="pl-k">+</span> t2 <span class="pl-k">+</span> t3 <span class="pl-c"><span class="pl-c">#</span> nm-3</span></td>
      </tr>
      <tr>
        <td id="L879" class="blob-num js-line-number" data-line-number="879"></td>
        <td id="LC879" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">6</span> <span class="pl-k">*</span> xiterm <span class="pl-k">/</span> (pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()) <span class="pl-c"><span class="pl-c">#</span> uses n_den instead of segden</span></td>
      </tr>
      <tr>
        <td id="L880" class="blob-num js-line-number" data-line-number="880"></td>
        <td id="LC880" class="blob-code blob-code-inner js-file-line">                                                  <span class="pl-c"><span class="pl-c">#</span> because it&#39;s just gonna multiply segratio again</span></td>
      </tr>
      <tr>
        <td id="L881" class="blob-num js-line-number" data-line-number="881"></td>
        <td id="LC881" class="blob-code blob-code-inner js-file-line">                                                  <span class="pl-c"><span class="pl-c">#</span> so cancel it out</span></td>
      </tr>
      <tr>
        <td id="L882" class="blob-num js-line-number" data-line-number="882"></td>
        <td id="LC882" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L883" class="blob-num js-line-number" data-line-number="883"></td>
        <td id="LC883" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L884" class="blob-num js-line-number" data-line-number="884"></td>
        <td id="LC884" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_1</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L885" class="blob-num js-line-number" data-line-number="885"></td>
        <td id="LC885" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L886" class="blob-num js-line-number" data-line-number="886"></td>
        <td id="LC886" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1/kT * cgss * sum of sum of _g1/cgss _g2/cgss * a1kl</span></td>
      </tr>
      <tr>
        <td id="L887" class="blob-num js-line-number" data-line-number="887"></td>
        <td id="LC887" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L888" class="blob-num js-line-number" data-line-number="888"></td>
        <td id="LC888" class="blob-code blob-code-inner js-file-line">        a1sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L889" class="blob-num js-line-number" data-line-number="889"></td>
        <td id="LC889" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L890" class="blob-num js-line-number" data-line-number="890"></td>
        <td id="LC890" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L891" class="blob-num js-line-number" data-line-number="891"></td>
        <td id="LC891" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L892" class="blob-num js-line-number" data-line-number="892"></td>
        <td id="LC892" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L893" class="blob-num js-line-number" data-line-number="893"></td>
        <td id="LC893" class="blob-code blob-code-inner js-file-line">                a1kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_1kl(c1.index, c2.index)</td>
      </tr>
      <tr>
        <td id="L894" class="blob-num js-line-number" data-line-number="894"></td>
        <td id="LC894" class="blob-code blob-code-inner js-file-line">                a1sum <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> a1kl </td>
      </tr>
      <tr>
        <td id="L895" class="blob-num js-line-number" data-line-number="895"></td>
        <td id="LC895" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> (cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> a1sum </td>
      </tr>
      <tr>
        <td id="L896" class="blob-num js-line-number" data-line-number="896"></td>
        <td id="LC896" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L897" class="blob-num js-line-number" data-line-number="897"></td>
        <td id="LC897" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L898" class="blob-num js-line-number" data-line-number="898"></td>
        <td id="LC898" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L899" class="blob-num js-line-number" data-line-number="899"></td>
        <td id="LC899" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L900" class="blob-num js-line-number" data-line-number="900"></td>
        <td id="LC900" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Ckl * [ x0kl^att(as1kl(att) + bkl(att)) - x0kl^rep(as1kl(rep) + bkl(rep)) ]</span></td>
      </tr>
      <tr>
        <td id="L901" class="blob-num js-line-number" data-line-number="901"></td>
        <td id="LC901" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L902" class="blob-num js-line-number" data-line-number="902"></td>
        <td id="LC902" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L903" class="blob-num js-line-number" data-line-number="903"></td>
        <td id="LC903" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L904" class="blob-num js-line-number" data-line-number="904"></td>
        <td id="LC904" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L905" class="blob-num js-line-number" data-line-number="905"></td>
        <td id="LC905" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L906" class="blob-num js-line-number" data-line-number="906"></td>
        <td id="LC906" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> comb.rep</td>
      </tr>
      <tr>
        <td id="L907" class="blob-num js-line-number" data-line-number="907"></td>
        <td id="LC907" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> comb.att</td>
      </tr>
      <tr>
        <td id="L908" class="blob-num js-line-number" data-line-number="908"></td>
        <td id="LC908" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> comb.premie()</td>
      </tr>
      <tr>
        <td id="L909" class="blob-num js-line-number" data-line-number="909"></td>
        <td id="LC909" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L910" class="blob-num js-line-number" data-line-number="910"></td>
        <td id="LC910" class="blob-code blob-code-inner js-file-line">        as1kl_a <span class="pl-k">=</span> <span class="pl-c1">self</span>.__as1kl(c1i, c2i, att)</td>
      </tr>
      <tr>
        <td id="L911" class="blob-num js-line-number" data-line-number="911"></td>
        <td id="LC911" class="blob-code blob-code-inner js-file-line">        bkl_a <span class="pl-k">=</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, att)</td>
      </tr>
      <tr>
        <td id="L912" class="blob-num js-line-number" data-line-number="912"></td>
        <td id="LC912" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L913" class="blob-num js-line-number" data-line-number="913"></td>
        <td id="LC913" class="blob-code blob-code-inner js-file-line">        as1kl_r <span class="pl-k">=</span> <span class="pl-c1">self</span>.__as1kl(c1i, c2i, rep)</td>
      </tr>
      <tr>
        <td id="L914" class="blob-num js-line-number" data-line-number="914"></td>
        <td id="LC914" class="blob-code blob-code-inner js-file-line">        bkl_r <span class="pl-k">=</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, rep)</td>
      </tr>
      <tr>
        <td id="L915" class="blob-num js-line-number" data-line-number="915"></td>
        <td id="LC915" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L916" class="blob-num js-line-number" data-line-number="916"></td>
        <td id="LC916" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, att) <span class="pl-k">*</span> (as1kl_a <span class="pl-k">+</span> bkl_a)</td>
      </tr>
      <tr>
        <td id="L917" class="blob-num js-line-number" data-line-number="917"></td>
        <td id="LC917" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, rep) <span class="pl-k">*</span> (as1kl_r <span class="pl-k">+</span> bkl_r)</td>
      </tr>
      <tr>
        <td id="L918" class="blob-num js-line-number" data-line-number="918"></td>
        <td id="LC918" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> ckl <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2)</td>
      </tr>
      <tr>
        <td id="L919" class="blob-num js-line-number" data-line-number="919"></td>
        <td id="LC919" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L920" class="blob-num js-line-number" data-line-number="920"></td>
        <td id="LC920" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L921" class="blob-num js-line-number" data-line-number="921"></td>
        <td id="LC921" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L922" class="blob-num js-line-number" data-line-number="922"></td>
        <td id="LC922" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L923" class="blob-num js-line-number" data-line-number="923"></td>
        <td id="LC923" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L924" class="blob-num js-line-number" data-line-number="924"></td>
        <td id="LC924" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1/kT)**2 * cgss * sum of sum of _g1/cgss _g2/cgss * a2kl</span></td>
      </tr>
      <tr>
        <td id="L925" class="blob-num js-line-number" data-line-number="925"></td>
        <td id="LC925" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L926" class="blob-num js-line-number" data-line-number="926"></td>
        <td id="LC926" class="blob-code blob-code-inner js-file-line">        a2sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L927" class="blob-num js-line-number" data-line-number="927"></td>
        <td id="LC927" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L928" class="blob-num js-line-number" data-line-number="928"></td>
        <td id="LC928" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L929" class="blob-num js-line-number" data-line-number="929"></td>
        <td id="LC929" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L930" class="blob-num js-line-number" data-line-number="930"></td>
        <td id="LC930" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L931" class="blob-num js-line-number" data-line-number="931"></td>
        <td id="LC931" class="blob-code blob-code-inner js-file-line">                a2kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_2kl(c1.index, c2.index)</td>
      </tr>
      <tr>
        <td id="L932" class="blob-num js-line-number" data-line-number="932"></td>
        <td id="LC932" class="blob-code blob-code-inner js-file-line">                a2sum <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> a2kl </td>
      </tr>
      <tr>
        <td id="L933" class="blob-num js-line-number" data-line-number="933"></td>
        <td id="LC933" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> <span class="pl-c1">pow</span>((cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature),<span class="pl-c1">2</span>) <span class="pl-k">*</span> a2sum </td>
      </tr>
      <tr>
        <td id="L934" class="blob-num js-line-number" data-line-number="934"></td>
        <td id="LC934" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L935" class="blob-num js-line-number" data-line-number="935"></td>
        <td id="LC935" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L936" class="blob-num js-line-number" data-line-number="936"></td>
        <td id="LC936" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_2kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L937" class="blob-num js-line-number" data-line-number="937"></td>
        <td id="LC937" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L938" class="blob-num js-line-number" data-line-number="938"></td>
        <td id="LC938" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1/2 * khs * (1-corrf) * epsikl * ckl^2 *</span></td>
      </tr>
      <tr>
        <td id="L939" class="blob-num js-line-number" data-line-number="939"></td>
        <td id="LC939" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        { x0kl^(2att) * (as1kl(2*att) + bkl(2*att))</span></td>
      </tr>
      <tr>
        <td id="L940" class="blob-num js-line-number" data-line-number="940"></td>
        <td id="LC940" class="blob-code blob-code-inner js-file-line"><span class="pl-s">          - 2*x0kl^(att+rep) * (as1kl(att+rep) + bkl(att+rep))</span></td>
      </tr>
      <tr>
        <td id="L941" class="blob-num js-line-number" data-line-number="941"></td>
        <td id="LC941" class="blob-code blob-code-inner js-file-line"><span class="pl-s">          + x0kl^(2rep) * (as1kl(2rep) + bkl(2rep)) }</span></td>
      </tr>
      <tr>
        <td id="L942" class="blob-num js-line-number" data-line-number="942"></td>
        <td id="LC942" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L943" class="blob-num js-line-number" data-line-number="943"></td>
        <td id="LC943" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L944" class="blob-num js-line-number" data-line-number="944"></td>
        <td id="LC944" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L945" class="blob-num js-line-number" data-line-number="945"></td>
        <td id="LC945" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L946" class="blob-num js-line-number" data-line-number="946"></td>
        <td id="LC946" class="blob-code blob-code-inner js-file-line">        corrf <span class="pl-k">=</span> <span class="pl-c1">self</span>.__corrf(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L947" class="blob-num js-line-number" data-line-number="947"></td>
        <td id="LC947" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L948" class="blob-num js-line-number" data-line-number="948"></td>
        <td id="LC948" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> comb.premie()</td>
      </tr>
      <tr>
        <td id="L949" class="blob-num js-line-number" data-line-number="949"></td>
        <td id="LC949" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L950" class="blob-num js-line-number" data-line-number="950"></td>
        <td id="LC950" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> comb.rep</td>
      </tr>
      <tr>
        <td id="L951" class="blob-num js-line-number" data-line-number="951"></td>
        <td id="LC951" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> comb.att</td>
      </tr>
      <tr>
        <td id="L952" class="blob-num js-line-number" data-line-number="952"></td>
        <td id="LC952" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L953" class="blob-num js-line-number" data-line-number="953"></td>
        <td id="LC953" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L954" class="blob-num js-line-number" data-line-number="954"></td>
        <td id="LC954" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L955" class="blob-num js-line-number" data-line-number="955"></td>
        <td id="LC955" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L956" class="blob-num js-line-number" data-line-number="956"></td>
        <td id="LC956" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L957" class="blob-num js-line-number" data-line-number="957"></td>
        <td id="LC957" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">0.5</span> <span class="pl-k">*</span> khs <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">+</span>corrf) <span class="pl-k">*</span> epsikl <span class="pl-k">*</span> <span class="pl-c1">pow</span>(ckl, <span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L958" class="blob-num js-line-number" data-line-number="958"></td>
        <td id="LC958" class="blob-code blob-code-inner js-file-line">        res <span class="pl-k">=</span> t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3</td>
      </tr>
      <tr>
        <td id="L959" class="blob-num js-line-number" data-line-number="959"></td>
        <td id="LC959" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L960" class="blob-num js-line-number" data-line-number="960"></td>
        <td id="LC960" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> fac <span class="pl-k">*</span> res</td>
      </tr>
      <tr>
        <td id="L961" class="blob-num js-line-number" data-line-number="961"></td>
        <td id="LC961" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L962" class="blob-num js-line-number" data-line-number="962"></td>
        <td id="LC962" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L963" class="blob-num js-line-number" data-line-number="963"></td>
        <td id="LC963" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__khs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L964" class="blob-num js-line-number" data-line-number="964"></td>
        <td id="LC964" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L965" class="blob-num js-line-number" data-line-number="965"></td>
        <td id="LC965" class="blob-code blob-code-inner js-file-line">        num <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix, <span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L966" class="blob-num js-line-number" data-line-number="966"></td>
        <td id="LC966" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span>xix <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">-</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>) <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L967" class="blob-num js-line-number" data-line-number="967"></td>
        <td id="LC967" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> num<span class="pl-k">/</span>den</td>
      </tr>
      <tr>
        <td id="L968" class="blob-num js-line-number" data-line-number="968"></td>
        <td id="LC968" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L969" class="blob-num js-line-number" data-line-number="969"></td>
        <td id="LC969" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__corrf</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L970" class="blob-num js-line-number" data-line-number="970"></td>
        <td id="LC970" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L971" class="blob-num js-line-number" data-line-number="971"></td>
        <td id="LC971" class="blob-code blob-code-inner js-file-line">        alkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L972" class="blob-num js-line-number" data-line-number="972"></td>
        <td id="LC972" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L973" class="blob-num js-line-number" data-line-number="973"></td>
        <td id="LC973" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">1</span>) <span class="pl-k">*</span> xisx</td>
      </tr>
      <tr>
        <td id="L974" class="blob-num js-line-number" data-line-number="974"></td>
        <td id="LC974" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">2</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">5</span>)</td>
      </tr>
      <tr>
        <td id="L975" class="blob-num js-line-number" data-line-number="975"></td>
        <td id="LC975" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">8</span>)</td>
      </tr>
      <tr>
        <td id="L976" class="blob-num js-line-number" data-line-number="976"></td>
        <td id="LC976" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> t1 <span class="pl-k">+</span> t2 <span class="pl-k">+</span> t3</td>
      </tr>
      <tr>
        <td id="L977" class="blob-num js-line-number" data-line-number="977"></td>
        <td id="LC977" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L978" class="blob-num js-line-number" data-line-number="978"></td>
        <td id="LC978" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L979" class="blob-num js-line-number" data-line-number="979"></td>
        <td id="LC979" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__alphakl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L980" class="blob-num js-line-number" data-line-number="980"></td>
        <td id="LC980" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L981" class="blob-num js-line-number" data-line-number="981"></td>
        <td id="LC981" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (comb.att <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L982" class="blob-num js-line-number" data-line-number="982"></td>
        <td id="LC982" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (comb.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L983" class="blob-num js-line-number" data-line-number="983"></td>
        <td id="LC983" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> comb.premie() <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2) </td>
      </tr>
      <tr>
        <td id="L984" class="blob-num js-line-number" data-line-number="984"></td>
        <td id="LC984" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L985" class="blob-num js-line-number" data-line-number="985"></td>
        <td id="LC985" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L986" class="blob-num js-line-number" data-line-number="986"></td>
        <td id="LC986" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L987" class="blob-num js-line-number" data-line-number="987"></td>
        <td id="LC987" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_3</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L988" class="blob-num js-line-number" data-line-number="988"></td>
        <td id="LC988" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L989" class="blob-num js-line-number" data-line-number="989"></td>
        <td id="LC989" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1/kT)**3 * cgss * sum of sum of _g1/cgss _g2/cgss * a3kl</span></td>
      </tr>
      <tr>
        <td id="L990" class="blob-num js-line-number" data-line-number="990"></td>
        <td id="LC990" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L991" class="blob-num js-line-number" data-line-number="991"></td>
        <td id="LC991" class="blob-code blob-code-inner js-file-line">        a3sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L992" class="blob-num js-line-number" data-line-number="992"></td>
        <td id="LC992" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L993" class="blob-num js-line-number" data-line-number="993"></td>
        <td id="LC993" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L994" class="blob-num js-line-number" data-line-number="994"></td>
        <td id="LC994" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L995" class="blob-num js-line-number" data-line-number="995"></td>
        <td id="LC995" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L996" class="blob-num js-line-number" data-line-number="996"></td>
        <td id="LC996" class="blob-code blob-code-inner js-file-line">                a3kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_3kl(c1.index, c2.index)</td>
      </tr>
      <tr>
        <td id="L997" class="blob-num js-line-number" data-line-number="997"></td>
        <td id="LC997" class="blob-code blob-code-inner js-file-line">                a3sum <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> a3kl </td>
      </tr>
      <tr>
        <td id="L998" class="blob-num js-line-number" data-line-number="998"></td>
        <td id="LC998" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> <span class="pl-c1">pow</span>((cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature),<span class="pl-c1">3</span>) <span class="pl-k">*</span> a3sum </td>
      </tr>
      <tr>
        <td id="L999" class="blob-num js-line-number" data-line-number="999"></td>
        <td id="LC999" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1000" class="blob-num js-line-number" data-line-number="1000"></td>
        <td id="LC1000" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1001" class="blob-num js-line-number" data-line-number="1001"></td>
        <td id="LC1001" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_3kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1002" class="blob-num js-line-number" data-line-number="1002"></td>
        <td id="LC1002" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L1003" class="blob-num js-line-number" data-line-number="1003"></td>
        <td id="LC1003" class="blob-code blob-code-inner js-file-line">        alkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1004" class="blob-num js-line-number" data-line-number="1004"></td>
        <td id="LC1004" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1005" class="blob-num js-line-number" data-line-number="1005"></td>
        <td id="LC1005" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1006" class="blob-num js-line-number" data-line-number="1006"></td>
        <td id="LC1006" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1007" class="blob-num js-line-number" data-line-number="1007"></td>
        <td id="LC1007" class="blob-code blob-code-inner js-file-line">        preexp <span class="pl-k">=</span> <span class="pl-k">-</span> <span class="pl-c1">pow</span>(epsikl,<span class="pl-c1">3</span>) <span class="pl-k">*</span> mt.f_m(alkl, <span class="pl-c1">4</span>) <span class="pl-k">*</span> xisx</td>
      </tr>
      <tr>
        <td id="L1008" class="blob-num js-line-number" data-line-number="1008"></td>
        <td id="LC1008" class="blob-code blob-code-inner js-file-line">        expt1 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">5</span>) <span class="pl-k">*</span> xisx </td>
      </tr>
      <tr>
        <td id="L1009" class="blob-num js-line-number" data-line-number="1009"></td>
        <td id="LC1009" class="blob-code blob-code-inner js-file-line">        expt2 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">6</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1010" class="blob-num js-line-number" data-line-number="1010"></td>
        <td id="LC1010" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1011" class="blob-num js-line-number" data-line-number="1011"></td>
        <td id="LC1011" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> preexp <span class="pl-k">*</span> exp(expt1 <span class="pl-k">+</span> expt2)</td>
      </tr>
      <tr>
        <td id="L1012" class="blob-num js-line-number" data-line-number="1012"></td>
        <td id="LC1012" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1013" class="blob-num js-line-number" data-line-number="1013"></td>
        <td id="LC1013" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1014" class="blob-num js-line-number" data-line-number="1014"></td>
        <td id="LC1014" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for chain</span></td>
      </tr>
      <tr>
        <td id="L1015" class="blob-num js-line-number" data-line-number="1015"></td>
        <td id="LC1015" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__gmieij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1016" class="blob-num js-line-number" data-line-number="1016"></td>
        <td id="LC1016" class="blob-code blob-code-inner js-file-line">        gdhs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__gdhs(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1017" class="blob-num js-line-number" data-line-number="1017"></td>
        <td id="LC1017" class="blob-code blob-code-inner js-file-line">        g1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g1(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1018" class="blob-num js-line-number" data-line-number="1018"></td>
        <td id="LC1018" class="blob-code blob-code-inner js-file-line">        g2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g2(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1019" class="blob-num js-line-number" data-line-number="1019"></td>
        <td id="LC1019" class="blob-code blob-code-inner js-file-line">        b <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature)</td>
      </tr>
      <tr>
        <td id="L1020" class="blob-num js-line-number" data-line-number="1020"></td>
        <td id="LC1020" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1021" class="blob-num js-line-number" data-line-number="1021"></td>
        <td id="LC1021" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1022" class="blob-num js-line-number" data-line-number="1022"></td>
        <td id="LC1022" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1023" class="blob-num js-line-number" data-line-number="1023"></td>
        <td id="LC1023" class="blob-code blob-code-inner js-file-line">        expt <span class="pl-k">=</span> b <span class="pl-k">*</span> epsi <span class="pl-k">*</span> g1 <span class="pl-k">/</span> gdhs <span class="pl-k">+</span> <span class="pl-c1">pow</span>(b<span class="pl-k">*</span>epsi,<span class="pl-c1">2</span>) <span class="pl-k">*</span> g2 <span class="pl-k">/</span> gdhs</td>
      </tr>
      <tr>
        <td id="L1024" class="blob-num js-line-number" data-line-number="1024"></td>
        <td id="LC1024" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> gdhs <span class="pl-k">*</span> exp(expt)</td>
      </tr>
      <tr>
        <td id="L1025" class="blob-num js-line-number" data-line-number="1025"></td>
        <td id="LC1025" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1026" class="blob-num js-line-number" data-line-number="1026"></td>
        <td id="LC1026" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1027" class="blob-num js-line-number" data-line-number="1027"></td>
        <td id="LC1027" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_xi_x</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1028" class="blob-num js-line-number" data-line-number="1028"></td>
        <td id="LC1028" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L1029" class="blob-num js-line-number" data-line-number="1029"></td>
        <td id="LC1029" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1030" class="blob-num js-line-number" data-line-number="1030"></td>
        <td id="LC1030" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1031" class="blob-num js-line-number" data-line-number="1031"></td>
        <td id="LC1031" class="blob-code blob-code-inner js-file-line">            c1i <span class="pl-k">=</span> c1.index</td>
      </tr>
      <tr>
        <td id="L1032" class="blob-num js-line-number" data-line-number="1032"></td>
        <td id="LC1032" class="blob-code blob-code-inner js-file-line">            xsi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c1] <span class="pl-k">*</span> c1.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1033" class="blob-num js-line-number" data-line-number="1033"></td>
        <td id="LC1033" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> c2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1034" class="blob-num js-line-number" data-line-number="1034"></td>
        <td id="LC1034" class="blob-code blob-code-inner js-file-line">                c2i <span class="pl-k">=</span> c2.index</td>
      </tr>
      <tr>
        <td id="L1035" class="blob-num js-line-number" data-line-number="1035"></td>
        <td id="LC1035" class="blob-code blob-code-inner js-file-line">                xsj <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c2] <span class="pl-k">*</span> c2.ms <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1036" class="blob-num js-line-number" data-line-number="1036"></td>
        <td id="LC1036" class="blob-code blob-code-inner js-file-line">                dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1037" class="blob-num js-line-number" data-line-number="1037"></td>
        <td id="LC1037" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsi <span class="pl-k">*</span> xsj <span class="pl-k">*</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1038" class="blob-num js-line-number" data-line-number="1038"></td>
        <td id="LC1038" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L1039" class="blob-num js-line-number" data-line-number="1039"></td>
        <td id="LC1039" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1040" class="blob-num js-line-number" data-line-number="1040"></td>
        <td id="LC1040" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_as1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1041" class="blob-num js-line-number" data-line-number="1041"></td>
        <td id="LC1041" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1042" class="blob-num js-line-number" data-line-number="1042"></td>
        <td id="LC1042" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        -2 * pi * (ep*d^3/(lam-3)) * (1/(1-xieff)^3) * (1 - xieff/2 + segden * derxieff * (3*(1-xieff/2)/(1-xieff) - 1/2))</span></td>
      </tr>
      <tr>
        <td id="L1043" class="blob-num js-line-number" data-line-number="1043"></td>
        <td id="LC1043" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1044" class="blob-num js-line-number" data-line-number="1044"></td>
        <td id="LC1044" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1045" class="blob-num js-line-number" data-line-number="1045"></td>
        <td id="LC1045" class="blob-code blob-code-inner js-file-line">        dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1046" class="blob-num js-line-number" data-line-number="1046"></td>
        <td id="LC1046" class="blob-code blob-code-inner js-file-line">        d_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1047" class="blob-num js-line-number" data-line-number="1047"></td>
        <td id="LC1047" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1048" class="blob-num js-line-number" data-line-number="1048"></td>
        <td id="LC1048" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1049" class="blob-num js-line-number" data-line-number="1049"></td>
        <td id="LC1049" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1050" class="blob-num js-line-number" data-line-number="1050"></td>
        <td id="LC1050" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1051" class="blob-num js-line-number" data-line-number="1051"></td>
        <td id="LC1051" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1052" class="blob-num js-line-number" data-line-number="1052"></td>
        <td id="LC1052" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1053" class="blob-num js-line-number" data-line-number="1053"></td>
        <td id="LC1053" class="blob-code blob-code-inner js-file-line">        derxixeff <span class="pl-k">=</span> mt.der_xi_x_eff(xix, derxix, lam)</td>
      </tr>
      <tr>
        <td id="L1054" class="blob-num js-line-number" data-line-number="1054"></td>
        <td id="LC1054" class="blob-code blob-code-inner js-file-line">        xixeff <span class="pl-k">=</span> mt.xi_x_eff(xix, lam)</td>
      </tr>
      <tr>
        <td id="L1055" class="blob-num js-line-number" data-line-number="1055"></td>
        <td id="LC1055" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1056" class="blob-num js-line-number" data-line-number="1056"></td>
        <td id="LC1056" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> d_3 <span class="pl-k">/</span> (lam<span class="pl-k">-</span><span class="pl-c1">3</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1057" class="blob-num js-line-number" data-line-number="1057"></td>
        <td id="LC1057" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1058" class="blob-num js-line-number" data-line-number="1058"></td>
        <td id="LC1058" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> segden <span class="pl-k">*</span> derxixeff</td>
      </tr>
      <tr>
        <td id="L1059" class="blob-num js-line-number" data-line-number="1059"></td>
        <td id="LC1059" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff<span class="pl-k">/</span><span class="pl-c1">2</span>) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff)</td>
      </tr>
      <tr>
        <td id="L1060" class="blob-num js-line-number" data-line-number="1060"></td>
        <td id="LC1060" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1061" class="blob-num js-line-number" data-line-number="1061"></td>
        <td id="LC1061" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-k">-</span> fac <span class="pl-k">*</span> (t1 <span class="pl-k">+</span> t2 <span class="pl-k">*</span> (t3 <span class="pl-k">-</span> <span class="pl-c1">1</span><span class="pl-k">/</span><span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L1062" class="blob-num js-line-number" data-line-number="1062"></td>
        <td id="LC1062" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1063" class="blob-num js-line-number" data-line-number="1063"></td>
        <td id="LC1063" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1064" class="blob-num js-line-number" data-line-number="1064"></td>
        <td id="LC1064" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1065" class="blob-num js-line-number" data-line-number="1065"></td>
        <td id="LC1065" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_bkl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1066" class="blob-num js-line-number" data-line-number="1066"></td>
        <td id="LC1066" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1067" class="blob-num js-line-number" data-line-number="1067"></td>
        <td id="LC1067" class="blob-code blob-code-inner js-file-line">        dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1068" class="blob-num js-line-number" data-line-number="1068"></td>
        <td id="LC1068" class="blob-code blob-code-inner js-file-line">        d_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1069" class="blob-num js-line-number" data-line-number="1069"></td>
        <td id="LC1069" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1070" class="blob-num js-line-number" data-line-number="1070"></td>
        <td id="LC1070" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1071" class="blob-num js-line-number" data-line-number="1071"></td>
        <td id="LC1071" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1072" class="blob-num js-line-number" data-line-number="1072"></td>
        <td id="LC1072" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(dkl)</td>
      </tr>
      <tr>
        <td id="L1073" class="blob-num js-line-number" data-line-number="1073"></td>
        <td id="LC1073" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1074" class="blob-num js-line-number" data-line-number="1074"></td>
        <td id="LC1074" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1075" class="blob-num js-line-number" data-line-number="1075"></td>
        <td id="LC1075" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1076" class="blob-num js-line-number" data-line-number="1076"></td>
        <td id="LC1076" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> d_3 <span class="pl-k">/</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1077" class="blob-num js-line-number" data-line-number="1077"></td>
        <td id="LC1077" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1078" class="blob-num js-line-number" data-line-number="1078"></td>
        <td id="LC1078" class="blob-code blob-code-inner js-file-line">        t11 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xix<span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1079" class="blob-num js-line-number" data-line-number="1079"></td>
        <td id="LC1079" class="blob-code blob-code-inner js-file-line">        t12 <span class="pl-k">=</span> segden <span class="pl-k">*</span> derxix</td>
      </tr>
      <tr>
        <td id="L1080" class="blob-num js-line-number" data-line-number="1080"></td>
        <td id="LC1080" class="blob-code blob-code-inner js-file-line">        t13 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix<span class="pl-k">/</span><span class="pl-c1">2</span>) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix) <span class="pl-k">-</span> <span class="pl-c1">1</span><span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1081" class="blob-num js-line-number" data-line-number="1081"></td>
        <td id="LC1081" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> t11 <span class="pl-k">+</span> t12 <span class="pl-k">*</span> t13</td>
      </tr>
      <tr>
        <td id="L1082" class="blob-num js-line-number" data-line-number="1082"></td>
        <td id="LC1082" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1083" class="blob-num js-line-number" data-line-number="1083"></td>
        <td id="LC1083" class="blob-code blob-code-inner js-file-line">        t21 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1084" class="blob-num js-line-number" data-line-number="1084"></td>
        <td id="LC1084" class="blob-code blob-code-inner js-file-line">        t22 <span class="pl-k">=</span> segden<span class="pl-k">*</span>derxix</td>
      </tr>
      <tr>
        <td id="L1085" class="blob-num js-line-number" data-line-number="1085"></td>
        <td id="LC1085" class="blob-code blob-code-inner js-file-line">        t23 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> xix <span class="pl-k">+</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> xix <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">+</span>xix) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix) </td>
      </tr>
      <tr>
        <td id="L1086" class="blob-num js-line-number" data-line-number="1086"></td>
        <td id="LC1086" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">9</span><span class="pl-k">/</span><span class="pl-c1">2</span> <span class="pl-k">*</span> (t21 <span class="pl-k">+</span> t22 <span class="pl-k">*</span> t23)</td>
      </tr>
      <tr>
        <td id="L1087" class="blob-num js-line-number" data-line-number="1087"></td>
        <td id="LC1087" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1088" class="blob-num js-line-number" data-line-number="1088"></td>
        <td id="LC1088" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> fac <span class="pl-k">*</span> (t1 <span class="pl-k">*</span> mt.intI(x0ii,lam) <span class="pl-k">-</span> t2 <span class="pl-k">*</span> mt.intJ(x0ii,lam))</td>
      </tr>
      <tr>
        <td id="L1089" class="blob-num js-line-number" data-line-number="1089"></td>
        <td id="LC1089" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1090" class="blob-num js-line-number" data-line-number="1090"></td>
        <td id="LC1090" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1091" class="blob-num js-line-number" data-line-number="1091"></td>
        <td id="LC1091" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_a1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1092" class="blob-num js-line-number" data-line-number="1092"></td>
        <td id="LC1092" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1093" class="blob-num js-line-number" data-line-number="1093"></td>
        <td id="LC1093" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(<span class="pl-c1">self</span>.hsd[c1i, c2i])</td>
      </tr>
      <tr>
        <td id="L1094" class="blob-num js-line-number" data-line-number="1094"></td>
        <td id="LC1094" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1095" class="blob-num js-line-number" data-line-number="1095"></td>
        <td id="LC1095" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1096" class="blob-num js-line-number" data-line-number="1096"></td>
        <td id="LC1096" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1097" class="blob-num js-line-number" data-line-number="1097"></td>
        <td id="LC1097" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1098" class="blob-num js-line-number" data-line-number="1098"></td>
        <td id="LC1098" class="blob-code blob-code-inner js-file-line">        tatt <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_as1kl(c1i, c2i, att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(c1i, c2i, att)</td>
      </tr>
      <tr>
        <td id="L1099" class="blob-num js-line-number" data-line-number="1099"></td>
        <td id="LC1099" class="blob-code blob-code-inner js-file-line">        trep <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_as1kl(c1i, c2i, rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(c1i, c2i, rep)</td>
      </tr>
      <tr>
        <td id="L1100" class="blob-num js-line-number" data-line-number="1100"></td>
        <td id="LC1100" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1101" class="blob-num js-line-number" data-line-number="1101"></td>
        <td id="LC1101" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> ckl <span class="pl-k">*</span> (<span class="pl-c1">pow</span>(x0ii, att) <span class="pl-k">*</span> tatt <span class="pl-k">-</span> <span class="pl-c1">pow</span>(x0ii, rep) <span class="pl-k">*</span> trep)</td>
      </tr>
      <tr>
        <td id="L1102" class="blob-num js-line-number" data-line-number="1102"></td>
        <td id="LC1102" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1103" class="blob-num js-line-number" data-line-number="1103"></td>
        <td id="LC1103" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1104" class="blob-num js-line-number" data-line-number="1104"></td>
        <td id="LC1104" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__gdhs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1105" class="blob-num js-line-number" data-line-number="1105"></td>
        <td id="LC1105" class="blob-code blob-code-inner js-file-line">        xi_x <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1106" class="blob-num js-line-number" data-line-number="1106"></td>
        <td id="LC1106" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1107" class="blob-num js-line-number" data-line-number="1107"></td>
        <td id="LC1107" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> comb.x0kl(<span class="pl-c1">self</span>.hsd[c1i, c2i])</td>
      </tr>
      <tr>
        <td id="L1108" class="blob-num js-line-number" data-line-number="1108"></td>
        <td id="LC1108" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1109" class="blob-num js-line-number" data-line-number="1109"></td>
        <td id="LC1109" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> mt.gdhs(x0ii, xi_x)</td>
      </tr>
      <tr>
        <td id="L1110" class="blob-num js-line-number" data-line-number="1110"></td>
        <td id="LC1110" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1111" class="blob-num js-line-number" data-line-number="1111"></td>
        <td id="LC1111" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1112" class="blob-num js-line-number" data-line-number="1112"></td>
        <td id="LC1112" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g1</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1113" class="blob-num js-line-number" data-line-number="1113"></td>
        <td id="LC1113" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1114" class="blob-num js-line-number" data-line-number="1114"></td>
        <td id="LC1114" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1 / (2pi epsiii hsdii^3) * [ 3 da1kl/dp - premieii * attii * x0ii^attii * (as1kl(attii) + Bkl(attii))/segden </span></td>
      </tr>
      <tr>
        <td id="L1115" class="blob-num js-line-number" data-line-number="1115"></td>
        <td id="LC1115" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                                     +  premieii * repii * x0ii^repii * (as1kl(repii) + Bkl(repii))/segden ]</span></td>
      </tr>
      <tr>
        <td id="L1116" class="blob-num js-line-number" data-line-number="1116"></td>
        <td id="LC1116" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1117" class="blob-num js-line-number" data-line-number="1117"></td>
        <td id="LC1117" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1118" class="blob-num js-line-number" data-line-number="1118"></td>
        <td id="LC1118" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1119" class="blob-num js-line-number" data-line-number="1119"></td>
        <td id="LC1119" class="blob-code blob-code-inner js-file-line">        premie <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1120" class="blob-num js-line-number" data-line-number="1120"></td>
        <td id="LC1120" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1121" class="blob-num js-line-number" data-line-number="1121"></td>
        <td id="LC1121" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1122" class="blob-num js-line-number" data-line-number="1122"></td>
        <td id="LC1122" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1123" class="blob-num js-line-number" data-line-number="1123"></td>
        <td id="LC1123" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1124" class="blob-num js-line-number" data-line-number="1124"></td>
        <td id="LC1124" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1125" class="blob-num js-line-number" data-line-number="1125"></td>
        <td id="LC1125" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1126" class="blob-num js-line-number" data-line-number="1126"></td>
        <td id="LC1126" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1127" class="blob-num js-line-number" data-line-number="1127"></td>
        <td id="LC1127" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">3</span><span class="pl-k">*</span> <span class="pl-c1">self</span>.__der_a1kl(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1128" class="blob-num js-line-number" data-line-number="1128"></td>
        <td id="LC1128" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> premie <span class="pl-k">*</span> att <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i,att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i,att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1129" class="blob-num js-line-number" data-line-number="1129"></td>
        <td id="LC1129" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> premie <span class="pl-k">*</span> rep <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i,rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i,rep)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1130" class="blob-num js-line-number" data-line-number="1130"></td>
        <td id="LC1130" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1131" class="blob-num js-line-number" data-line-number="1131"></td>
        <td id="LC1131" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (<span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(hsd,<span class="pl-c1">3</span>)) <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3)</td>
      </tr>
      <tr>
        <td id="L1132" class="blob-num js-line-number" data-line-number="1132"></td>
        <td id="LC1132" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1133" class="blob-num js-line-number" data-line-number="1133"></td>
        <td id="LC1133" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1134" class="blob-num js-line-number" data-line-number="1134"></td>
        <td id="LC1134" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1135" class="blob-num js-line-number" data-line-number="1135"></td>
        <td id="LC1135" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1136" class="blob-num js-line-number" data-line-number="1136"></td>
        <td id="LC1136" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1137" class="blob-num js-line-number" data-line-number="1137"></td>
        <td id="LC1137" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1+gammacii) * g2MCA(hsdii)</span></td>
      </tr>
      <tr>
        <td id="L1138" class="blob-num js-line-number" data-line-number="1138"></td>
        <td id="LC1138" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1139" class="blob-num js-line-number" data-line-number="1139"></td>
        <td id="LC1139" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1140" class="blob-num js-line-number" data-line-number="1140"></td>
        <td id="LC1140" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L1141" class="blob-num js-line-number" data-line-number="1141"></td>
        <td id="LC1141" class="blob-code blob-code-inner js-file-line">        alii <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1142" class="blob-num js-line-number" data-line-number="1142"></td>
        <td id="LC1142" class="blob-code blob-code-inner js-file-line">        theta <span class="pl-k">=</span> exp(gcomp.epsilon <span class="pl-k">/</span> <span class="pl-c1">self</span>.temperature) <span class="pl-k">-</span><span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L1143" class="blob-num js-line-number" data-line-number="1143"></td>
        <td id="LC1143" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1144" class="blob-num js-line-number" data-line-number="1144"></td>
        <td id="LC1144" class="blob-code blob-code-inner js-file-line">        gammacii <span class="pl-k">=</span> cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">0</span>] <span class="pl-k">*</span> (<span class="pl-k">-</span>tanh(cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">1</span>] <span class="pl-k">*</span> (cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">2</span>]<span class="pl-k">-</span>alii)) <span class="pl-k">+</span> <span class="pl-c1">1</span>) <span class="pl-k">*</span> xisx <span class="pl-k">*</span> theta <span class="pl-k">*</span> exp(cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">3</span>]<span class="pl-k">*</span>xisx <span class="pl-k">+</span> cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">4</span>] <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L1145" class="blob-num js-line-number" data-line-number="1145"></td>
        <td id="LC1145" class="blob-code blob-code-inner js-file-line">        g2mca <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g2mca(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1146" class="blob-num js-line-number" data-line-number="1146"></td>
        <td id="LC1146" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> (<span class="pl-c1">1</span> <span class="pl-k">+</span> gammacii) <span class="pl-k">*</span> g2mca</td>
      </tr>
      <tr>
        <td id="L1147" class="blob-num js-line-number" data-line-number="1147"></td>
        <td id="LC1147" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1148" class="blob-num js-line-number" data-line-number="1148"></td>
        <td id="LC1148" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1149" class="blob-num js-line-number" data-line-number="1149"></td>
        <td id="LC1149" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_khs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1150" class="blob-num js-line-number" data-line-number="1150"></td>
        <td id="LC1150" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1151" class="blob-num js-line-number" data-line-number="1151"></td>
        <td id="LC1151" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1152" class="blob-num js-line-number" data-line-number="1152"></td>
        <td id="LC1152" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1153" class="blob-num js-line-number" data-line-number="1153"></td>
        <td id="LC1153" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span>xix <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">-</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>) <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L1154" class="blob-num js-line-number" data-line-number="1154"></td>
        <td id="LC1154" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">4</span> <span class="pl-k">*</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">3</span>) <span class="pl-k">/</span> den</td>
      </tr>
      <tr>
        <td id="L1155" class="blob-num js-line-number" data-line-number="1155"></td>
        <td id="LC1155" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">4</span>) <span class="pl-k">*</span> (<span class="pl-c1">4</span> <span class="pl-k">+</span> <span class="pl-c1">8</span><span class="pl-k">*</span>xix <span class="pl-k">-</span><span class="pl-c1">12</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(den,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1156" class="blob-num js-line-number" data-line-number="1156"></td>
        <td id="LC1156" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> derxix <span class="pl-k">*</span> <span class="pl-k">-</span>(t1 <span class="pl-k">+</span> t2)</td>
      </tr>
      <tr>
        <td id="L1157" class="blob-num js-line-number" data-line-number="1157"></td>
        <td id="LC1157" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1158" class="blob-num js-line-number" data-line-number="1158"></td>
        <td id="LC1158" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1159" class="blob-num js-line-number" data-line-number="1159"></td>
        <td id="LC1159" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1160" class="blob-num js-line-number" data-line-number="1160"></td>
        <td id="LC1160" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_a2kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1161" class="blob-num js-line-number" data-line-number="1161"></td>
        <td id="LC1161" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L1162" class="blob-num js-line-number" data-line-number="1162"></td>
        <td id="LC1162" class="blob-code blob-code-inner js-file-line">        derkhs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_khs()</td>
      </tr>
      <tr>
        <td id="L1163" class="blob-num js-line-number" data-line-number="1163"></td>
        <td id="LC1163" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1164" class="blob-num js-line-number" data-line-number="1164"></td>
        <td id="LC1164" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1165" class="blob-num js-line-number" data-line-number="1165"></td>
        <td id="LC1165" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1166" class="blob-num js-line-number" data-line-number="1166"></td>
        <td id="LC1166" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1167" class="blob-num js-line-number" data-line-number="1167"></td>
        <td id="LC1167" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1168" class="blob-num js-line-number" data-line-number="1168"></td>
        <td id="LC1168" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1169" class="blob-num js-line-number" data-line-number="1169"></td>
        <td id="LC1169" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1170" class="blob-num js-line-number" data-line-number="1170"></td>
        <td id="LC1170" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1171" class="blob-num js-line-number" data-line-number="1171"></td>
        <td id="LC1171" class="blob-code blob-code-inner js-file-line">        t11 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L1172" class="blob-num js-line-number" data-line-number="1172"></td>
        <td id="LC1172" class="blob-code blob-code-inner js-file-line">        t12 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L1173" class="blob-num js-line-number" data-line-number="1173"></td>
        <td id="LC1173" class="blob-code blob-code-inner js-file-line">        t13 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L1174" class="blob-num js-line-number" data-line-number="1174"></td>
        <td id="LC1174" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span>  derkhs <span class="pl-k">*</span> (t11 <span class="pl-k">-</span> t12 <span class="pl-k">+</span> t13)</td>
      </tr>
      <tr>
        <td id="L1175" class="blob-num js-line-number" data-line-number="1175"></td>
        <td id="LC1175" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1176" class="blob-num js-line-number" data-line-number="1176"></td>
        <td id="LC1176" class="blob-code blob-code-inner js-file-line">        t21 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L1177" class="blob-num js-line-number" data-line-number="1177"></td>
        <td id="LC1177" class="blob-code blob-code-inner js-file-line">        t22 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(c1i, c2i, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(c1i, c2i, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L1178" class="blob-num js-line-number" data-line-number="1178"></td>
        <td id="LC1178" class="blob-code blob-code-inner js-file-line">        t23 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(c1i, c2i, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L1179" class="blob-num js-line-number" data-line-number="1179"></td>
        <td id="LC1179" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> khs <span class="pl-k">*</span> (t21 <span class="pl-k">-</span> t22 <span class="pl-k">+</span> t23)</td>
      </tr>
      <tr>
        <td id="L1180" class="blob-num js-line-number" data-line-number="1180"></td>
        <td id="LC1180" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1181" class="blob-num js-line-number" data-line-number="1181"></td>
        <td id="LC1181" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0.5</span> <span class="pl-k">*</span> epsi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(ckl, <span class="pl-c1">2</span>) <span class="pl-k">*</span> (t1 <span class="pl-k">+</span> t2)</td>
      </tr>
      <tr>
        <td id="L1182" class="blob-num js-line-number" data-line-number="1182"></td>
        <td id="LC1182" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1183" class="blob-num js-line-number" data-line-number="1183"></td>
        <td id="LC1183" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1184" class="blob-num js-line-number" data-line-number="1184"></td>
        <td id="LC1184" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g2mca</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">c1i</span>, <span class="pl-smi">c2i</span>):</td>
      </tr>
      <tr>
        <td id="L1185" class="blob-num js-line-number" data-line-number="1185"></td>
        <td id="LC1185" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1186" class="blob-num js-line-number" data-line-number="1186"></td>
        <td id="LC1186" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1 / (2pi epsiii^2 hsdii^3) *</span></td>
      </tr>
      <tr>
        <td id="L1187" class="blob-num js-line-number" data-line-number="1187"></td>
        <td id="LC1187" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        [</span></td>
      </tr>
      <tr>
        <td id="L1188" class="blob-num js-line-number" data-line-number="1188"></td>
        <td id="LC1188" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        3 * d/dp (a2ii/(1+chi))</span></td>
      </tr>
      <tr>
        <td id="L1189" class="blob-num js-line-number" data-line-number="1189"></td>
        <td id="LC1189" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - epsiii * KHS * premie^2 * rep * x0ii^2rep * (as1kl(2rep) + Bkl(2rep))/segden </span></td>
      </tr>
      <tr>
        <td id="L1190" class="blob-num js-line-number" data-line-number="1190"></td>
        <td id="LC1190" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        + epsiii * KHS * premie^2 * (rep+att) * x0ii^(rep+att) * (as1kl(rep+att) + Bkl(rep+att))/segden </span></td>
      </tr>
      <tr>
        <td id="L1191" class="blob-num js-line-number" data-line-number="1191"></td>
        <td id="LC1191" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - epsiii * KHS * premie^2 * att * x0ii^2att * (as1kl(2att) + Bkl(2att))/segden </span></td>
      </tr>
      <tr>
        <td id="L1192" class="blob-num js-line-number" data-line-number="1192"></td>
        <td id="LC1192" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        ]</span></td>
      </tr>
      <tr>
        <td id="L1193" class="blob-num js-line-number" data-line-number="1193"></td>
        <td id="LC1193" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1194" class="blob-num js-line-number" data-line-number="1194"></td>
        <td id="LC1194" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1195" class="blob-num js-line-number" data-line-number="1195"></td>
        <td id="LC1195" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.hsd[c1i, c2i]</td>
      </tr>
      <tr>
        <td id="L1196" class="blob-num js-line-number" data-line-number="1196"></td>
        <td id="LC1196" class="blob-code blob-code-inner js-file-line">        premie <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1197" class="blob-num js-line-number" data-line-number="1197"></td>
        <td id="LC1197" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1198" class="blob-num js-line-number" data-line-number="1198"></td>
        <td id="LC1198" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1199" class="blob-num js-line-number" data-line-number="1199"></td>
        <td id="LC1199" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1200" class="blob-num js-line-number" data-line-number="1200"></td>
        <td id="LC1200" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1201" class="blob-num js-line-number" data-line-number="1201"></td>
        <td id="LC1201" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L1202" class="blob-num js-line-number" data-line-number="1202"></td>
        <td id="LC1202" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1203" class="blob-num js-line-number" data-line-number="1203"></td>
        <td id="LC1203" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1204" class="blob-num js-line-number" data-line-number="1204"></td>
        <td id="LC1204" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1205" class="blob-num js-line-number" data-line-number="1205"></td>
        <td id="LC1205" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.__der_a2kl(c1i, c2i)</td>
      </tr>
      <tr>
        <td id="L1206" class="blob-num js-line-number" data-line-number="1206"></td>
        <td id="LC1206" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> rep <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i,<span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i,<span class="pl-c1">2</span><span class="pl-k">*</span>rep)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1207" class="blob-num js-line-number" data-line-number="1207"></td>
        <td id="LC1207" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> (rep<span class="pl-k">+</span>att) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, rep<span class="pl-k">+</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i,rep<span class="pl-k">+</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i,rep<span class="pl-k">+</span>att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1208" class="blob-num js-line-number" data-line-number="1208"></td>
        <td id="LC1208" class="blob-code blob-code-inner js-file-line">        t4 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> att <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(c1i, c2i,<span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(c1i, c2i,<span class="pl-c1">2</span><span class="pl-k">*</span>att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1209" class="blob-num js-line-number" data-line-number="1209"></td>
        <td id="LC1209" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L1210" class="blob-num js-line-number" data-line-number="1210"></td>
        <td id="LC1210" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (<span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(epsi,<span class="pl-c1">2</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(hsd, <span class="pl-c1">3</span>)) <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3 <span class="pl-k">-</span> t4)</td>
      </tr>
      <tr>
        <td id="L1211" class="blob-num js-line-number" data-line-number="1211"></td>
        <td id="LC1211" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1212" class="blob-num js-line-number" data-line-number="1212"></td>
        <td id="LC1212" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1213" class="blob-num js-line-number" data-line-number="1213"></td>
        <td id="LC1213" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1214" class="blob-num js-line-number" data-line-number="1214"></td>
        <td id="LC1214" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for assoc</span></td>
      </tr>
      <tr>
        <td id="L1215" class="blob-num js-line-number" data-line-number="1215"></td>
        <td id="LC1215" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1216" class="blob-num js-line-number" data-line-number="1216"></td>
        <td id="LC1216" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">SAFTgMieSystem</span>(<span class="pl-e">System</span>):</td>
      </tr>
      <tr>
        <td id="L1217" class="blob-num js-line-number" data-line-number="1217"></td>
        <td id="LC1217" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1218" class="blob-num js-line-number" data-line-number="1218"></td>
        <td id="LC1218" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    SAFT-gamma Mie System as defined by SAFT-gamma Mie Equation-of-state</span></td>
      </tr>
      <tr>
        <td id="L1219" class="blob-num js-line-number" data-line-number="1219"></td>
        <td id="LC1219" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455</span></td>
      </tr>
      <tr>
        <td id="L1220" class="blob-num js-line-number" data-line-number="1220"></td>
        <td id="LC1220" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1221" class="blob-num js-line-number" data-line-number="1221"></td>
        <td id="LC1221" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span><span class="pl-k">=</span><span class="pl-c1">293</span>, <span class="pl-smi">pressure</span><span class="pl-k">=</span><span class="pl-c1">101325</span>, <span class="pl-smi">volume</span><span class="pl-k">=</span><span class="pl-c1">1000</span>):</td>
      </tr>
      <tr>
        <td id="L1222" class="blob-num js-line-number" data-line-number="1222"></td>
        <td id="LC1222" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">super</span>().<span class="pl-c1">__init__</span>(temperature, pressure, volume)</td>
      </tr>
      <tr>
        <td id="L1223" class="blob-num js-line-number" data-line-number="1223"></td>
        <td id="LC1223" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.groups <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L1224" class="blob-num js-line-number" data-line-number="1224"></td>
        <td id="LC1224" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1225" class="blob-num js-line-number" data-line-number="1225"></td>
        <td id="LC1225" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> stored values</span></td>
      </tr>
      <tr>
        <td id="L1226" class="blob-num js-line-number" data-line-number="1226"></td>
        <td id="LC1226" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ghsd <span class="pl-k">=</span> np.array([[<span class="pl-c1">None</span>]])</td>
      </tr>
      <tr>
        <td id="L1227" class="blob-num js-line-number" data-line-number="1227"></td>
        <td id="LC1227" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1228" class="blob-num js-line-number" data-line-number="1228"></td>
        <td id="LC1228" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1229" class="blob-num js-line-number" data-line-number="1229"></td>
        <td id="LC1229" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.array([[<span class="pl-c1">None</span>]])</td>
      </tr>
      <tr>
        <td id="L1230" class="blob-num js-line-number" data-line-number="1230"></td>
        <td id="LC1230" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.chsd <span class="pl-k">=</span> np.array([<span class="pl-c1">None</span>])</td>
      </tr>
      <tr>
        <td id="L1231" class="blob-num js-line-number" data-line-number="1231"></td>
        <td id="LC1231" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ccomb <span class="pl-k">=</span> np.array([<span class="pl-c1">None</span>])</td>
      </tr>
      <tr>
        <td id="L1232" class="blob-num js-line-number" data-line-number="1232"></td>
        <td id="LC1232" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L1233" class="blob-num js-line-number" data-line-number="1233"></td>
        <td id="LC1233" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_groups</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1234" class="blob-num js-line-number" data-line-number="1234"></td>
        <td id="LC1234" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.groups <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L1235" class="blob-num js-line-number" data-line-number="1235"></td>
        <td id="LC1235" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1236" class="blob-num js-line-number" data-line-number="1236"></td>
        <td id="LC1236" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g <span class="pl-k">in</span> comp.groups:</td>
      </tr>
      <tr>
        <td id="L1237" class="blob-num js-line-number" data-line-number="1237"></td>
        <td id="LC1237" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> g <span class="pl-k">not</span> <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1238" class="blob-num js-line-number" data-line-number="1238"></td>
        <td id="LC1238" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.groups[g] <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">*</span> comp.groups[g] </td>
      </tr>
      <tr>
        <td id="L1239" class="blob-num js-line-number" data-line-number="1239"></td>
        <td id="LC1239" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L1240" class="blob-num js-line-number" data-line-number="1240"></td>
        <td id="LC1240" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.groups[g] <span class="pl-k">+=</span> <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">*</span> comp.groups[g]</td>
      </tr>
      <tr>
        <td id="L1241" class="blob-num js-line-number" data-line-number="1241"></td>
        <td id="LC1241" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.groups</td>
      </tr>
      <tr>
        <td id="L1242" class="blob-num js-line-number" data-line-number="1242"></td>
        <td id="LC1242" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1243" class="blob-num js-line-number" data-line-number="1243"></td>
        <td id="LC1243" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">quick_set</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-k">*</span><span class="pl-smi">args</span>):</td>
      </tr>
      <tr>
        <td id="L1244" class="blob-num js-line-number" data-line-number="1244"></td>
        <td id="LC1244" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">super</span>().quick_set(<span class="pl-k">*</span>args)</td>
      </tr>
      <tr>
        <td id="L1245" class="blob-num js-line-number" data-line-number="1245"></td>
        <td id="LC1245" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.get_groups()</td>
      </tr>
      <tr>
        <td id="L1246" class="blob-num js-line-number" data-line-number="1246"></td>
        <td id="LC1246" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L1247" class="blob-num js-line-number" data-line-number="1247"></td>
        <td id="LC1247" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1248" class="blob-num js-line-number" data-line-number="1248"></td>
        <td id="LC1248" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">add_comp</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comp</span>, <span class="pl-smi">number</span>):</td>
      </tr>
      <tr>
        <td id="L1249" class="blob-num js-line-number" data-line-number="1249"></td>
        <td id="LC1249" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">super</span>().quick_set(<span class="pl-k">*</span>args)</td>
      </tr>
      <tr>
        <td id="L1250" class="blob-num js-line-number" data-line-number="1250"></td>
        <td id="LC1250" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.get_groups()</td>
      </tr>
      <tr>
        <td id="L1251" class="blob-num js-line-number" data-line-number="1251"></td>
        <td id="LC1251" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L1252" class="blob-num js-line-number" data-line-number="1252"></td>
        <td id="LC1252" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1253" class="blob-num js-line-number" data-line-number="1253"></td>
        <td id="LC1253" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1254" class="blob-num js-line-number" data-line-number="1254"></td>
        <td id="LC1254" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L1255" class="blob-num js-line-number" data-line-number="1255"></td>
        <td id="LC1255" class="blob-code blob-code-inner js-file-line">        nmg <span class="pl-k">=</span> GMieGroup.n_total()</td>
      </tr>
      <tr>
        <td id="L1256" class="blob-num js-line-number" data-line-number="1256"></td>
        <td id="LC1256" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ghsd <span class="pl-k">=</span> np.zeros((nmg, nmg), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1257" class="blob-num js-line-number" data-line-number="1257"></td>
        <td id="LC1257" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.zeros((nmg, nmg), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1258" class="blob-num js-line-number" data-line-number="1258"></td>
        <td id="LC1258" class="blob-code blob-code-inner js-file-line">        segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1259" class="blob-num js-line-number" data-line-number="1259"></td>
        <td id="LC1259" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1260" class="blob-num js-line-number" data-line-number="1260"></td>
        <td id="LC1260" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1261" class="blob-num js-line-number" data-line-number="1261"></td>
        <td id="LC1261" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L1262" class="blob-num js-line-number" data-line-number="1262"></td>
        <td id="LC1262" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L1263" class="blob-num js-line-number" data-line-number="1263"></td>
        <td id="LC1263" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1264" class="blob-num js-line-number" data-line-number="1264"></td>
        <td id="LC1264" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.get_groups() <span class="pl-c"><span class="pl-c">#</span> in case molfrac is Var</span></td>
      </tr>
      <tr>
        <td id="L1265" class="blob-num js-line-number" data-line-number="1265"></td>
        <td id="LC1265" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1266" class="blob-num js-line-number" data-line-number="1266"></td>
        <td id="LC1266" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1267" class="blob-num js-line-number" data-line-number="1267"></td>
        <td id="LC1267" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> hard-sphere diam</span></td>
      </tr>
      <tr>
        <td id="L1268" class="blob-num js-line-number" data-line-number="1268"></td>
        <td id="LC1268" class="blob-code blob-code-inner js-file-line">            gi <span class="pl-k">=</span> g.index</td>
      </tr>
      <tr>
        <td id="L1269" class="blob-num js-line-number" data-line-number="1269"></td>
        <td id="LC1269" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ghsd[gi, gi] <span class="pl-k">=</span> g.hsd(<span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> <span class="pl-c1">1e-10</span></td>
      </tr>
      <tr>
        <td id="L1270" class="blob-num js-line-number" data-line-number="1270"></td>
        <td id="LC1270" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1271" class="blob-num js-line-number" data-line-number="1271"></td>
        <td id="LC1271" class="blob-code blob-code-inner js-file-line">                gi2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1272" class="blob-num js-line-number" data-line-number="1272"></td>
        <td id="LC1272" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.ghsd[gi, gi2] <span class="pl-k">==</span> <span class="pl-c1">0</span>. <span class="pl-k">and</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2] <span class="pl-k">!=</span> <span class="pl-c1">0</span>.:</td>
      </tr>
      <tr>
        <td id="L1273" class="blob-num js-line-number" data-line-number="1273"></td>
        <td id="LC1273" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.ghsd[gi, gi2] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ghsd[gi,gi] <span class="pl-k">+</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1274" class="blob-num js-line-number" data-line-number="1274"></td>
        <td id="LC1274" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.ghsd[gi2, gi] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ghsd[gi,gi] <span class="pl-k">+</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1275" class="blob-num js-line-number" data-line-number="1275"></td>
        <td id="LC1275" class="blob-code blob-code-inner js-file-line">                <span class="pl-c"><span class="pl-c">#</span>gcomb</span></td>
      </tr>
      <tr>
        <td id="L1276" class="blob-num js-line-number" data-line-number="1276"></td>
        <td id="LC1276" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.gcomb[gi, gi2] <span class="pl-k">=</span> g <span class="pl-k">+</span> g2</td>
      </tr>
      <tr>
        <td id="L1277" class="blob-num js-line-number" data-line-number="1277"></td>
        <td id="LC1277" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> segden</span></td>
      </tr>
      <tr>
        <td id="L1278" class="blob-num js-line-number" data-line-number="1278"></td>
        <td id="LC1278" class="blob-code blob-code-inner js-file-line">            segratio <span class="pl-k">+=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.vk <span class="pl-k">*</span> g.sf</td>
      </tr>
      <tr>
        <td id="L1279" class="blob-num js-line-number" data-line-number="1279"></td>
        <td id="LC1279" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1280" class="blob-num js-line-number" data-line-number="1280"></td>
        <td id="LC1280" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> segratio</td>
      </tr>
      <tr>
        <td id="L1281" class="blob-num js-line-number" data-line-number="1281"></td>
        <td id="LC1281" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> segratio <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L1282" class="blob-num js-line-number" data-line-number="1282"></td>
        <td id="LC1282" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1283" class="blob-num js-line-number" data-line-number="1283"></td>
        <td id="LC1283" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> chain comb</span></td>
      </tr>
      <tr>
        <td id="L1284" class="blob-num js-line-number" data-line-number="1284"></td>
        <td id="LC1284" class="blob-code blob-code-inner js-file-line">        nc <span class="pl-k">=</span> GMieComponent.n_total()</td>
      </tr>
      <tr>
        <td id="L1285" class="blob-num js-line-number" data-line-number="1285"></td>
        <td id="LC1285" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.chsd <span class="pl-k">=</span> np.zeros((nc,), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1286" class="blob-num js-line-number" data-line-number="1286"></td>
        <td id="LC1286" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ccomb <span class="pl-k">=</span> np.zeros((nc,), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1287" class="blob-num js-line-number" data-line-number="1287"></td>
        <td id="LC1287" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1288" class="blob-num js-line-number" data-line-number="1288"></td>
        <td id="LC1288" class="blob-code blob-code-inner js-file-line">            ci <span class="pl-k">=</span> c.index</td>
      </tr>
      <tr>
        <td id="L1289" class="blob-num js-line-number" data-line-number="1289"></td>
        <td id="LC1289" class="blob-code blob-code-inner js-file-line">            gc, hsdii <span class="pl-k">=</span> c.get_gtypeii(<span class="pl-v">hsd</span><span class="pl-k">=</span><span class="pl-c1">self</span>.ghsd)</td>
      </tr>
      <tr>
        <td id="L1290" class="blob-num js-line-number" data-line-number="1290"></td>
        <td id="LC1290" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ccomb[ci] <span class="pl-k">=</span> gc</td>
      </tr>
      <tr>
        <td id="L1291" class="blob-num js-line-number" data-line-number="1291"></td>
        <td id="LC1291" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.chsd[ci] <span class="pl-k">=</span> hsdii</td>
      </tr>
      <tr>
        <td id="L1292" class="blob-num js-line-number" data-line-number="1292"></td>
        <td id="LC1292" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1293" class="blob-num js-line-number" data-line-number="1293"></td>
        <td id="LC1293" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L1294" class="blob-num js-line-number" data-line-number="1294"></td>
        <td id="LC1294" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_ideal() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_res() <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L1295" class="blob-num js-line-number" data-line-number="1295"></td>
        <td id="LC1295" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1296" class="blob-num js-line-number" data-line-number="1296"></td>
        <td id="LC1296" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L1297" class="blob-num js-line-number" data-line-number="1297"></td>
        <td id="LC1297" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1298" class="blob-num js-line-number" data-line-number="1298"></td>
        <td id="LC1298" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_residual</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1299" class="blob-num js-line-number" data-line-number="1299"></td>
        <td id="LC1299" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L1300" class="blob-num js-line-number" data-line-number="1300"></td>
        <td id="LC1300" class="blob-code blob-code-inner js-file-line">        nmg <span class="pl-k">=</span> GMieGroup.n_total()</td>
      </tr>
      <tr>
        <td id="L1301" class="blob-num js-line-number" data-line-number="1301"></td>
        <td id="LC1301" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ghsd <span class="pl-k">=</span> np.zeros((nmg, nmg), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1302" class="blob-num js-line-number" data-line-number="1302"></td>
        <td id="LC1302" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.gcomb <span class="pl-k">=</span> np.zeros((nmg, nmg), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1303" class="blob-num js-line-number" data-line-number="1303"></td>
        <td id="LC1303" class="blob-code blob-code-inner js-file-line">        segratio <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1304" class="blob-num js-line-number" data-line-number="1304"></td>
        <td id="LC1304" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1305" class="blob-num js-line-number" data-line-number="1305"></td>
        <td id="LC1305" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1306" class="blob-num js-line-number" data-line-number="1306"></td>
        <td id="LC1306" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L1307" class="blob-num js-line-number" data-line-number="1307"></td>
        <td id="LC1307" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L1308" class="blob-num js-line-number" data-line-number="1308"></td>
        <td id="LC1308" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1309" class="blob-num js-line-number" data-line-number="1309"></td>
        <td id="LC1309" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.get_groups() <span class="pl-c"><span class="pl-c">#</span> in case molfrac is Var</span></td>
      </tr>
      <tr>
        <td id="L1310" class="blob-num js-line-number" data-line-number="1310"></td>
        <td id="LC1310" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1311" class="blob-num js-line-number" data-line-number="1311"></td>
        <td id="LC1311" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1312" class="blob-num js-line-number" data-line-number="1312"></td>
        <td id="LC1312" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> hard-sphere diam</span></td>
      </tr>
      <tr>
        <td id="L1313" class="blob-num js-line-number" data-line-number="1313"></td>
        <td id="LC1313" class="blob-code blob-code-inner js-file-line">            gi <span class="pl-k">=</span> g.index</td>
      </tr>
      <tr>
        <td id="L1314" class="blob-num js-line-number" data-line-number="1314"></td>
        <td id="LC1314" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ghsd[gi, gi] <span class="pl-k">=</span> g.hsd(<span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> <span class="pl-c1">1e-10</span></td>
      </tr>
      <tr>
        <td id="L1315" class="blob-num js-line-number" data-line-number="1315"></td>
        <td id="LC1315" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1316" class="blob-num js-line-number" data-line-number="1316"></td>
        <td id="LC1316" class="blob-code blob-code-inner js-file-line">                gi2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1317" class="blob-num js-line-number" data-line-number="1317"></td>
        <td id="LC1317" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">self</span>.ghsd[gi, gi2] <span class="pl-k">==</span> <span class="pl-c1">0</span>. <span class="pl-k">and</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2] <span class="pl-k">!=</span> <span class="pl-c1">0</span>.:</td>
      </tr>
      <tr>
        <td id="L1318" class="blob-num js-line-number" data-line-number="1318"></td>
        <td id="LC1318" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.ghsd[gi, gi2] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ghsd[gi,gi] <span class="pl-k">+</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1319" class="blob-num js-line-number" data-line-number="1319"></td>
        <td id="LC1319" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.ghsd[gi2, gi] <span class="pl-k">=</span> (<span class="pl-c1">self</span>.ghsd[gi,gi] <span class="pl-k">+</span> <span class="pl-c1">self</span>.ghsd[gi2, gi2]) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1320" class="blob-num js-line-number" data-line-number="1320"></td>
        <td id="LC1320" class="blob-code blob-code-inner js-file-line">                <span class="pl-c"><span class="pl-c">#</span>gcomb</span></td>
      </tr>
      <tr>
        <td id="L1321" class="blob-num js-line-number" data-line-number="1321"></td>
        <td id="LC1321" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.gcomb[gi, gi2] <span class="pl-k">=</span> g <span class="pl-k">+</span> g2</td>
      </tr>
      <tr>
        <td id="L1322" class="blob-num js-line-number" data-line-number="1322"></td>
        <td id="LC1322" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> segden</span></td>
      </tr>
      <tr>
        <td id="L1323" class="blob-num js-line-number" data-line-number="1323"></td>
        <td id="LC1323" class="blob-code blob-code-inner js-file-line">            segratio <span class="pl-k">+=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.vk <span class="pl-k">*</span> g.sf</td>
      </tr>
      <tr>
        <td id="L1324" class="blob-num js-line-number" data-line-number="1324"></td>
        <td id="LC1324" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1325" class="blob-num js-line-number" data-line-number="1325"></td>
        <td id="LC1325" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segratio <span class="pl-k">=</span> segratio</td>
      </tr>
      <tr>
        <td id="L1326" class="blob-num js-line-number" data-line-number="1326"></td>
        <td id="LC1326" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.segden <span class="pl-k">=</span> segratio <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L1327" class="blob-num js-line-number" data-line-number="1327"></td>
        <td id="LC1327" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1328" class="blob-num js-line-number" data-line-number="1328"></td>
        <td id="LC1328" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> chain comb</span></td>
      </tr>
      <tr>
        <td id="L1329" class="blob-num js-line-number" data-line-number="1329"></td>
        <td id="LC1329" class="blob-code blob-code-inner js-file-line">        nc <span class="pl-k">=</span> GMieComponent.n_total()</td>
      </tr>
      <tr>
        <td id="L1330" class="blob-num js-line-number" data-line-number="1330"></td>
        <td id="LC1330" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.chsd <span class="pl-k">=</span> np.zeros((nc,), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1331" class="blob-num js-line-number" data-line-number="1331"></td>
        <td id="LC1331" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ccomb <span class="pl-k">=</span> np.zeros((nc,), <span class="pl-v">dtype</span><span class="pl-k">=</span><span class="pl-c1">object</span>)</td>
      </tr>
      <tr>
        <td id="L1332" class="blob-num js-line-number" data-line-number="1332"></td>
        <td id="LC1332" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1333" class="blob-num js-line-number" data-line-number="1333"></td>
        <td id="LC1333" class="blob-code blob-code-inner js-file-line">            ci <span class="pl-k">=</span> c.index</td>
      </tr>
      <tr>
        <td id="L1334" class="blob-num js-line-number" data-line-number="1334"></td>
        <td id="LC1334" class="blob-code blob-code-inner js-file-line">            gc, hsdii <span class="pl-k">=</span> c.get_gtypeii(<span class="pl-v">hsd</span><span class="pl-k">=</span><span class="pl-c1">self</span>.ghsd)</td>
      </tr>
      <tr>
        <td id="L1335" class="blob-num js-line-number" data-line-number="1335"></td>
        <td id="LC1335" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ccomb[ci] <span class="pl-k">=</span> gc</td>
      </tr>
      <tr>
        <td id="L1336" class="blob-num js-line-number" data-line-number="1336"></td>
        <td id="LC1336" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.chsd[ci] <span class="pl-k">=</span> hsdii</td>
      </tr>
      <tr>
        <td id="L1337" class="blob-num js-line-number" data-line-number="1337"></td>
        <td id="LC1337" class="blob-code blob-code-inner js-file-line">            </td>
      </tr>
      <tr>
        <td id="L1338" class="blob-num js-line-number" data-line-number="1338"></td>
        <td id="LC1338" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L1339" class="blob-num js-line-number" data-line-number="1339"></td>
        <td id="LC1339" class="blob-code blob-code-inner js-file-line">        a <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_res()</td>
      </tr>
      <tr>
        <td id="L1340" class="blob-num js-line-number" data-line-number="1340"></td>
        <td id="LC1340" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> a <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L1341" class="blob-num js-line-number" data-line-number="1341"></td>
        <td id="LC1341" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1342" class="blob-num js-line-number" data-line-number="1342"></td>
        <td id="LC1342" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L1343" class="blob-num js-line-number" data-line-number="1343"></td>
        <td id="LC1343" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1344" class="blob-num js-line-number" data-line-number="1344"></td>
        <td id="LC1344" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">helmholtz_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1345" class="blob-num js-line-number" data-line-number="1345"></td>
        <td id="LC1345" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate stored values</span></td>
      </tr>
      <tr>
        <td id="L1346" class="blob-num js-line-number" data-line-number="1346"></td>
        <td id="LC1346" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1347" class="blob-num js-line-number" data-line-number="1347"></td>
        <td id="LC1347" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> molfrac</span></td>
      </tr>
      <tr>
        <td id="L1348" class="blob-num js-line-number" data-line-number="1348"></td>
        <td id="LC1348" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.molfrac[comp] <span class="pl-k">=</span> <span class="pl-c1">self</span>.moles[comp] <span class="pl-k">/</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L1349" class="blob-num js-line-number" data-line-number="1349"></td>
        <td id="LC1349" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1350" class="blob-num js-line-number" data-line-number="1350"></td>
        <td id="LC1350" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.get_groups() <span class="pl-c"><span class="pl-c">#</span> in case molfrac is Var</span></td>
      </tr>
      <tr>
        <td id="L1351" class="blob-num js-line-number" data-line-number="1351"></td>
        <td id="LC1351" class="blob-code blob-code-inner js-file-line">            </td>
      </tr>
      <tr>
        <td id="L1352" class="blob-num js-line-number" data-line-number="1352"></td>
        <td id="LC1352" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> calculate helmholtz through indiv contributions</span></td>
      </tr>
      <tr>
        <td id="L1353" class="blob-num js-line-number" data-line-number="1353"></td>
        <td id="LC1353" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_ideal()</td>
      </tr>
      <tr>
        <td id="L1354" class="blob-num js-line-number" data-line-number="1354"></td>
        <td id="LC1354" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1355" class="blob-num js-line-number" data-line-number="1355"></td>
        <td id="LC1355" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> A</td>
      </tr>
      <tr>
        <td id="L1356" class="blob-num js-line-number" data-line-number="1356"></td>
        <td id="LC1356" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1357" class="blob-num js-line-number" data-line-number="1357"></td>
        <td id="LC1357" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1358" class="blob-num js-line-number" data-line-number="1358"></td>
        <td id="LC1358" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1359" class="blob-num js-line-number" data-line-number="1359"></td>
        <td id="LC1359" class="blob-code blob-code-inner js-file-line">        mole_tol <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules</td>
      </tr>
      <tr>
        <td id="L1360" class="blob-num js-line-number" data-line-number="1360"></td>
        <td id="LC1360" class="blob-code blob-code-inner js-file-line">        cpideal <span class="pl-k">=</span> <span class="pl-c1">True</span></td>
      </tr>
      <tr>
        <td id="L1361" class="blob-num js-line-number" data-line-number="1361"></td>
        <td id="LC1361" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1362" class="blob-num js-line-number" data-line-number="1362"></td>
        <td id="LC1362" class="blob-code blob-code-inner js-file-line">            cpint <span class="pl-k">=</span> comp.cp_int(<span class="pl-c1">self</span>.temperature)</td>
      </tr>
      <tr>
        <td id="L1363" class="blob-num js-line-number" data-line-number="1363"></td>
        <td id="LC1363" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> cpint <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L1364" class="blob-num js-line-number" data-line-number="1364"></td>
        <td id="LC1364" class="blob-code blob-code-inner js-file-line">                cpideal <span class="pl-k">=</span> <span class="pl-c1">False</span></td>
      </tr>
      <tr>
        <td id="L1365" class="blob-num js-line-number" data-line-number="1365"></td>
        <td id="LC1365" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1366" class="blob-num js-line-number" data-line-number="1366"></td>
        <td id="LC1366" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> cpideal <span class="pl-k">==</span> <span class="pl-c1">False</span>:</td>
      </tr>
      <tr>
        <td id="L1367" class="blob-num js-line-number" data-line-number="1367"></td>
        <td id="LC1367" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1368" class="blob-num js-line-number" data-line-number="1368"></td>
        <td id="LC1368" class="blob-code blob-code-inner js-file-line">                debrogv <span class="pl-k">=</span> <span class="pl-c1">pow</span>(comp.thdebroglie(<span class="pl-c1">self</span>.temperature),<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1369" class="blob-num js-line-number" data-line-number="1369"></td>
        <td id="LC1369" class="blob-code blob-code-inner js-file-line">                molfrac <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[comp]</td>
      </tr>
      <tr>
        <td id="L1370" class="blob-num js-line-number" data-line-number="1370"></td>
        <td id="LC1370" class="blob-code blob-code-inner js-file-line">                nden <span class="pl-k">=</span> molfrac <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()</td>
      </tr>
      <tr>
        <td id="L1371" class="blob-num js-line-number" data-line-number="1371"></td>
        <td id="LC1371" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">=</span> result <span class="pl-k">+</span> molfrac <span class="pl-k">*</span> log(nden <span class="pl-k">*</span> debrogv) </td>
      </tr>
      <tr>
        <td id="L1372" class="blob-num js-line-number" data-line-number="1372"></td>
        <td id="LC1372" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">=</span> result <span class="pl-k">-</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L1373" class="blob-num js-line-number" data-line-number="1373"></td>
        <td id="LC1373" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> result <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L1374" class="blob-num js-line-number" data-line-number="1374"></td>
        <td id="LC1374" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L1375" class="blob-num js-line-number" data-line-number="1375"></td>
        <td id="LC1375" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> comp <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1376" class="blob-num js-line-number" data-line-number="1376"></td>
        <td id="LC1376" class="blob-code blob-code-inner js-file-line">            h0 <span class="pl-k">=</span> comp.ref[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L1377" class="blob-num js-line-number" data-line-number="1377"></td>
        <td id="LC1377" class="blob-code blob-code-inner js-file-line">            s0 <span class="pl-k">=</span> comp.ref[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L1378" class="blob-num js-line-number" data-line-number="1378"></td>
        <td id="LC1378" class="blob-code blob-code-inner js-file-line">            tref <span class="pl-k">=</span> comp.ref[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L1379" class="blob-num js-line-number" data-line-number="1379"></td>
        <td id="LC1379" class="blob-code blob-code-inner js-file-line">            pref <span class="pl-k">=</span> comp.ref[<span class="pl-c1">3</span>]</td>
      </tr>
      <tr>
        <td id="L1380" class="blob-num js-line-number" data-line-number="1380"></td>
        <td id="LC1380" class="blob-code blob-code-inner js-file-line">            vref <span class="pl-k">=</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> tref <span class="pl-k">/</span> pref</td>
      </tr>
      <tr>
        <td id="L1381" class="blob-num js-line-number" data-line-number="1381"></td>
        <td id="LC1381" class="blob-code blob-code-inner js-file-line">            t <span class="pl-k">=</span> <span class="pl-c1">self</span>.temperature</td>
      </tr>
      <tr>
        <td id="L1382" class="blob-num js-line-number" data-line-number="1382"></td>
        <td id="LC1382" class="blob-code blob-code-inner js-file-line">            xi <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[comp]</td>
      </tr>
      <tr>
        <td id="L1383" class="blob-num js-line-number" data-line-number="1383"></td>
        <td id="LC1383" class="blob-code blob-code-inner js-file-line">            n_mol <span class="pl-k">=</span> <span class="pl-c1">self</span>.n_molecules <span class="pl-k">/</span> cst.Na</td>
      </tr>
      <tr>
        <td id="L1384" class="blob-num js-line-number" data-line-number="1384"></td>
        <td id="LC1384" class="blob-code blob-code-inner js-file-line">            V <span class="pl-k">=</span> <span class="pl-c1">self</span>.volume <span class="pl-k">*</span> cst.nmtom<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L1385" class="blob-num js-line-number" data-line-number="1385"></td>
        <td id="LC1385" class="blob-code blob-code-inner js-file-line">            cpint <span class="pl-k">=</span> comp.cp_int(t)</td>
      </tr>
      <tr>
        <td id="L1386" class="blob-num js-line-number" data-line-number="1386"></td>
        <td id="LC1386" class="blob-code blob-code-inner js-file-line">            cptint <span class="pl-k">=</span> comp.cp_t_int(t)</td>
      </tr>
      <tr>
        <td id="L1387" class="blob-num js-line-number" data-line-number="1387"></td>
        <td id="LC1387" class="blob-code blob-code-inner js-file-line">            </td>
      </tr>
      <tr>
        <td id="L1388" class="blob-num js-line-number" data-line-number="1388"></td>
        <td id="LC1388" class="blob-code blob-code-inner js-file-line">            si <span class="pl-k">=</span> cptint <span class="pl-k">-</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> log(xi <span class="pl-k">*</span> t <span class="pl-k">*</span> vref <span class="pl-k">*</span> n_mol <span class="pl-k">/</span> (tref<span class="pl-k">*</span>V)) <span class="pl-k">+</span> s0</td>
      </tr>
      <tr>
        <td id="L1389" class="blob-num js-line-number" data-line-number="1389"></td>
        <td id="LC1389" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1390" class="blob-num js-line-number" data-line-number="1390"></td>
        <td id="LC1390" class="blob-code blob-code-inner js-file-line">            ai_mol <span class="pl-k">=</span> cpint <span class="pl-k">-</span> t <span class="pl-k">*</span> si <span class="pl-k">-</span> cst.Na <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> t <span class="pl-k">+</span> h0 <span class="pl-c"><span class="pl-c">#</span> per mol</span></td>
      </tr>
      <tr>
        <td id="L1391" class="blob-num js-line-number" data-line-number="1391"></td>
        <td id="LC1391" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">=</span> result <span class="pl-k">+</span> xi <span class="pl-k">*</span> n_mol <span class="pl-k">*</span> ai_mol</td>
      </tr>
      <tr>
        <td id="L1392" class="blob-num js-line-number" data-line-number="1392"></td>
        <td id="LC1392" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1393" class="blob-num js-line-number" data-line-number="1393"></td>
        <td id="LC1393" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1394" class="blob-num js-line-number" data-line-number="1394"></td>
        <td id="LC1394" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1395" class="blob-num js-line-number" data-line-number="1395"></td>
        <td id="LC1395" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_res</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1396" class="blob-num js-line-number" data-line-number="1396"></td>
        <td id="LC1396" class="blob-code blob-code-inner js-file-line">        a_r <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_mono() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_chain() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_assoc()</td>
      </tr>
      <tr>
        <td id="L1397" class="blob-num js-line-number" data-line-number="1397"></td>
        <td id="LC1397" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1398" class="blob-num js-line-number" data-line-number="1398"></td>
        <td id="LC1398" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> a_r</td>
      </tr>
      <tr>
        <td id="L1399" class="blob-num js-line-number" data-line-number="1399"></td>
        <td id="LC1399" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1400" class="blob-num js-line-number" data-line-number="1400"></td>
        <td id="LC1400" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_mono</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1401" class="blob-num js-line-number" data-line-number="1401"></td>
        <td id="LC1401" class="blob-code blob-code-inner js-file-line">        a_m <span class="pl-k">=</span> <span class="pl-c1">self</span>.a_hs() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_1() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_2() <span class="pl-k">+</span> <span class="pl-c1">self</span>.a_3()</td>
      </tr>
      <tr>
        <td id="L1402" class="blob-num js-line-number" data-line-number="1402"></td>
        <td id="LC1402" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> a_m</td>
      </tr>
      <tr>
        <td id="L1403" class="blob-num js-line-number" data-line-number="1403"></td>
        <td id="LC1403" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1404" class="blob-num js-line-number" data-line-number="1404"></td>
        <td id="LC1404" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_chain</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1405" class="blob-num js-line-number" data-line-number="1405"></td>
        <td id="LC1405" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1406" class="blob-num js-line-number" data-line-number="1406"></td>
        <td id="LC1406" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> c <span class="pl-k">in</span> <span class="pl-c1">self</span>.moles:</td>
      </tr>
      <tr>
        <td id="L1407" class="blob-num js-line-number" data-line-number="1407"></td>
        <td id="LC1407" class="blob-code blob-code-inner js-file-line">            molfrac <span class="pl-k">=</span> <span class="pl-c1">self</span>.molfrac[c]</td>
      </tr>
      <tr>
        <td id="L1408" class="blob-num js-line-number" data-line-number="1408"></td>
        <td id="LC1408" class="blob-code blob-code-inner js-file-line">            gmieii <span class="pl-k">=</span> <span class="pl-c1">self</span>.__gmieij(c.index)</td>
      </tr>
      <tr>
        <td id="L1409" class="blob-num js-line-number" data-line-number="1409"></td>
        <td id="LC1409" class="blob-code blob-code-inner js-file-line">            mseq <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1410" class="blob-num js-line-number" data-line-number="1410"></td>
        <td id="LC1410" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g <span class="pl-k">in</span> c.groups:</td>
      </tr>
      <tr>
        <td id="L1411" class="blob-num js-line-number" data-line-number="1411"></td>
        <td id="LC1411" class="blob-code blob-code-inner js-file-line">                mseq <span class="pl-k">+=</span> c.groups[g] <span class="pl-k">*</span> g.vk <span class="pl-k">*</span> g.sf</td>
      </tr>
      <tr>
        <td id="L1412" class="blob-num js-line-number" data-line-number="1412"></td>
        <td id="LC1412" class="blob-code blob-code-inner js-file-line">            result <span class="pl-k">+=</span> molfrac <span class="pl-k">*</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> mseq) <span class="pl-k">*</span> log(gmieii)</td>
      </tr>
      <tr>
        <td id="L1413" class="blob-num js-line-number" data-line-number="1413"></td>
        <td id="LC1413" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1414" class="blob-num js-line-number" data-line-number="1414"></td>
        <td id="LC1414" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1415" class="blob-num js-line-number" data-line-number="1415"></td>
        <td id="LC1415" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_assoc</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1416" class="blob-num js-line-number" data-line-number="1416"></td>
        <td id="LC1416" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">0</span> <span class="pl-c"><span class="pl-c">#</span> placeholder self.volume ** 3 * 0.1 + exp(-self.volume)</span></td>
      </tr>
      <tr>
        <td id="L1417" class="blob-num js-line-number" data-line-number="1417"></td>
        <td id="LC1417" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1418" class="blob-num js-line-number" data-line-number="1418"></td>
        <td id="LC1418" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for all</span></td>
      </tr>
      <tr>
        <td id="L1419" class="blob-num js-line-number" data-line-number="1419"></td>
        <td id="LC1419" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_m</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">m</span>):</td>
      </tr>
      <tr>
        <td id="L1420" class="blob-num js-line-number" data-line-number="1420"></td>
        <td id="LC1420" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L1421" class="blob-num js-line-number" data-line-number="1421"></td>
        <td id="LC1421" class="blob-code blob-code-inner js-file-line">        gsum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1422" class="blob-num js-line-number" data-line-number="1422"></td>
        <td id="LC1422" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1423" class="blob-num js-line-number" data-line-number="1423"></td>
        <td id="LC1423" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.vk <span class="pl-k">*</span> g.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1424" class="blob-num js-line-number" data-line-number="1424"></td>
        <td id="LC1424" class="blob-code blob-code-inner js-file-line">            gsum <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> <span class="pl-c1">pow</span>(<span class="pl-c1">self</span>.ghsd[g.index, g.index], m)</td>
      </tr>
      <tr>
        <td id="L1425" class="blob-num js-line-number" data-line-number="1425"></td>
        <td id="LC1425" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> gsum</td>
      </tr>
      <tr>
        <td id="L1426" class="blob-num js-line-number" data-line-number="1426"></td>
        <td id="LC1426" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1427" class="blob-num js-line-number" data-line-number="1427"></td>
        <td id="LC1427" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_x</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1428" class="blob-num js-line-number" data-line-number="1428"></td>
        <td id="LC1428" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L1429" class="blob-num js-line-number" data-line-number="1429"></td>
        <td id="LC1429" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1430" class="blob-num js-line-number" data-line-number="1430"></td>
        <td id="LC1430" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1431" class="blob-num js-line-number" data-line-number="1431"></td>
        <td id="LC1431" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1432" class="blob-num js-line-number" data-line-number="1432"></td>
        <td id="LC1432" class="blob-code blob-code-inner js-file-line">            g1i <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L1433" class="blob-num js-line-number" data-line-number="1433"></td>
        <td id="LC1433" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1434" class="blob-num js-line-number" data-line-number="1434"></td>
        <td id="LC1434" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1435" class="blob-num js-line-number" data-line-number="1435"></td>
        <td id="LC1435" class="blob-code blob-code-inner js-file-line">                g2i <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1436" class="blob-num js-line-number" data-line-number="1436"></td>
        <td id="LC1436" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1437" class="blob-num js-line-number" data-line-number="1437"></td>
        <td id="LC1437" class="blob-code blob-code-inner js-file-line">                dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.ghsd[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1438" class="blob-num js-line-number" data-line-number="1438"></td>
        <td id="LC1438" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1439" class="blob-num js-line-number" data-line-number="1439"></td>
        <td id="LC1439" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L1440" class="blob-num js-line-number" data-line-number="1440"></td>
        <td id="LC1440" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1441" class="blob-num js-line-number" data-line-number="1441"></td>
        <td id="LC1441" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__xi_sx</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1442" class="blob-num js-line-number" data-line-number="1442"></td>
        <td id="LC1442" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.segden <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L1443" class="blob-num js-line-number" data-line-number="1443"></td>
        <td id="LC1443" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1444" class="blob-num js-line-number" data-line-number="1444"></td>
        <td id="LC1444" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1445" class="blob-num js-line-number" data-line-number="1445"></td>
        <td id="LC1445" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1446" class="blob-num js-line-number" data-line-number="1446"></td>
        <td id="LC1446" class="blob-code blob-code-inner js-file-line">            g1i <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L1447" class="blob-num js-line-number" data-line-number="1447"></td>
        <td id="LC1447" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1448" class="blob-num js-line-number" data-line-number="1448"></td>
        <td id="LC1448" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1449" class="blob-num js-line-number" data-line-number="1449"></td>
        <td id="LC1449" class="blob-code blob-code-inner js-file-line">                g2i <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1450" class="blob-num js-line-number" data-line-number="1450"></td>
        <td id="LC1450" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1451" class="blob-num js-line-number" data-line-number="1451"></td>
        <td id="LC1451" class="blob-code blob-code-inner js-file-line">                skl <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[g1i, g2i].sigma <span class="pl-k">*</span> cst.atom</td>
      </tr>
      <tr>
        <td id="L1452" class="blob-num js-line-number" data-line-number="1452"></td>
        <td id="LC1452" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> <span class="pl-c1">pow</span>(skl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1453" class="blob-num js-line-number" data-line-number="1453"></td>
        <td id="LC1453" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L1454" class="blob-num js-line-number" data-line-number="1454"></td>
        <td id="LC1454" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1455" class="blob-num js-line-number" data-line-number="1455"></td>
        <td id="LC1455" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__bkl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>, <span class="pl-smi">dkl</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1456" class="blob-num js-line-number" data-line-number="1456"></td>
        <td id="LC1456" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1457" class="blob-num js-line-number" data-line-number="1457"></td>
        <td id="LC1457" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Use 2 groups as input</span></td>
      </tr>
      <tr>
        <td id="L1458" class="blob-num js-line-number" data-line-number="1458"></td>
        <td id="LC1458" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        set rep = True for lam_r, rep = False for lam_a</span></td>
      </tr>
      <tr>
        <td id="L1459" class="blob-num js-line-number" data-line-number="1459"></td>
        <td id="LC1459" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        2 * pi * segden * dkl^3 epsikl * bkl_sum(lambda_r or a, xi_x)</span></td>
      </tr>
      <tr>
        <td id="L1460" class="blob-num js-line-number" data-line-number="1460"></td>
        <td id="LC1460" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1461" class="blob-num js-line-number" data-line-number="1461"></td>
        <td id="LC1461" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1462" class="blob-num js-line-number" data-line-number="1462"></td>
        <td id="LC1462" class="blob-code blob-code-inner js-file-line">        dkl_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1463" class="blob-num js-line-number" data-line-number="1463"></td>
        <td id="LC1463" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1464" class="blob-num js-line-number" data-line-number="1464"></td>
        <td id="LC1464" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1465" class="blob-num js-line-number" data-line-number="1465"></td>
        <td id="LC1465" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(dkl)</td>
      </tr>
      <tr>
        <td id="L1466" class="blob-num js-line-number" data-line-number="1466"></td>
        <td id="LC1466" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1467" class="blob-num js-line-number" data-line-number="1467"></td>
        <td id="LC1467" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1468" class="blob-num js-line-number" data-line-number="1468"></td>
        <td id="LC1468" class="blob-code blob-code-inner js-file-line">        bsum <span class="pl-k">=</span> mt.bkl_sum(xix, x0kl, lam)</td>
      </tr>
      <tr>
        <td id="L1469" class="blob-num js-line-number" data-line-number="1469"></td>
        <td id="LC1469" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span>pi<span class="pl-k">*</span>segden<span class="pl-k">*</span>dkl_3<span class="pl-k">*</span>epsikl<span class="pl-k">*</span>bsum</td>
      </tr>
      <tr>
        <td id="L1470" class="blob-num js-line-number" data-line-number="1470"></td>
        <td id="LC1470" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1471" class="blob-num js-line-number" data-line-number="1471"></td>
        <td id="LC1471" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1472" class="blob-num js-line-number" data-line-number="1472"></td>
        <td id="LC1472" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1473" class="blob-num js-line-number" data-line-number="1473"></td>
        <td id="LC1473" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__as1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>, <span class="pl-smi">dkl</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1474" class="blob-num js-line-number" data-line-number="1474"></td>
        <td id="LC1474" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1475" class="blob-num js-line-number" data-line-number="1475"></td>
        <td id="LC1475" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - 2 * pi * segden * (epsikl * dkl^3/ (lam - 3)) * (1 - xi_x_eff/2)/(1-xi_x_eff)^3</span></td>
      </tr>
      <tr>
        <td id="L1476" class="blob-num js-line-number" data-line-number="1476"></td>
        <td id="LC1476" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1477" class="blob-num js-line-number" data-line-number="1477"></td>
        <td id="LC1477" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1478" class="blob-num js-line-number" data-line-number="1478"></td>
        <td id="LC1478" class="blob-code blob-code-inner js-file-line">        dkl_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1479" class="blob-num js-line-number" data-line-number="1479"></td>
        <td id="LC1479" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1480" class="blob-num js-line-number" data-line-number="1480"></td>
        <td id="LC1480" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1481" class="blob-num js-line-number" data-line-number="1481"></td>
        <td id="LC1481" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1482" class="blob-num js-line-number" data-line-number="1482"></td>
        <td id="LC1482" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1483" class="blob-num js-line-number" data-line-number="1483"></td>
        <td id="LC1483" class="blob-code blob-code-inner js-file-line">        xixeff <span class="pl-k">=</span> mt.xi_x_eff(xix, lam)</td>
      </tr>
      <tr>
        <td id="L1484" class="blob-num js-line-number" data-line-number="1484"></td>
        <td id="LC1484" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1485" class="blob-num js-line-number" data-line-number="1485"></td>
        <td id="LC1485" class="blob-code blob-code-inner js-file-line">        num <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff<span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1486" class="blob-num js-line-number" data-line-number="1486"></td>
        <td id="LC1486" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1487" class="blob-num js-line-number" data-line-number="1487"></td>
        <td id="LC1487" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1488" class="blob-num js-line-number" data-line-number="1488"></td>
        <td id="LC1488" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-k">-</span><span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> segden <span class="pl-k">*</span> (epsikl <span class="pl-k">*</span> dkl_3 <span class="pl-k">/</span> (lam <span class="pl-k">-</span> <span class="pl-c1">3</span>)) <span class="pl-k">*</span> (num<span class="pl-k">/</span>den)</td>
      </tr>
      <tr>
        <td id="L1489" class="blob-num js-line-number" data-line-number="1489"></td>
        <td id="LC1489" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1490" class="blob-num js-line-number" data-line-number="1490"></td>
        <td id="LC1490" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1491" class="blob-num js-line-number" data-line-number="1491"></td>
        <td id="LC1491" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1492" class="blob-num js-line-number" data-line-number="1492"></td>
        <td id="LC1492" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for mono</span></td>
      </tr>
      <tr>
        <td id="L1493" class="blob-num js-line-number" data-line-number="1493"></td>
        <td id="LC1493" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_hs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1494" class="blob-num js-line-number" data-line-number="1494"></td>
        <td id="LC1494" class="blob-code blob-code-inner js-file-line">        xi1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">1</span>) <span class="pl-c"><span class="pl-c">#</span> nm-2</span></td>
      </tr>
      <tr>
        <td id="L1495" class="blob-num js-line-number" data-line-number="1495"></td>
        <td id="LC1495" class="blob-code blob-code-inner js-file-line">        xi2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">2</span>) <span class="pl-c"><span class="pl-c">#</span> nm-1</span></td>
      </tr>
      <tr>
        <td id="L1496" class="blob-num js-line-number" data-line-number="1496"></td>
        <td id="LC1496" class="blob-code blob-code-inner js-file-line">        xi3 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">3</span>) <span class="pl-c"><span class="pl-c">#</span> dimless</span></td>
      </tr>
      <tr>
        <td id="L1497" class="blob-num js-line-number" data-line-number="1497"></td>
        <td id="LC1497" class="blob-code blob-code-inner js-file-line">        xi0 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_m(<span class="pl-c1">0</span>) <span class="pl-c"><span class="pl-c">#</span> nm-3</span></td>
      </tr>
      <tr>
        <td id="L1498" class="blob-num js-line-number" data-line-number="1498"></td>
        <td id="LC1498" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1499" class="blob-num js-line-number" data-line-number="1499"></td>
        <td id="LC1499" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">len</span>(<span class="pl-c1">self</span>.moles) <span class="pl-k">==</span> <span class="pl-c1">1</span>:</td>
      </tr>
      <tr>
        <td id="L1500" class="blob-num js-line-number" data-line-number="1500"></td>
        <td id="LC1500" class="blob-code blob-code-inner js-file-line">            t1 <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L1501" class="blob-num js-line-number" data-line-number="1501"></td>
        <td id="LC1501" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L1502" class="blob-num js-line-number" data-line-number="1502"></td>
        <td id="LC1502" class="blob-code blob-code-inner js-file-line">            t1 <span class="pl-k">=</span> (<span class="pl-c1">pow</span>(xi2, <span class="pl-c1">3</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(xi3, <span class="pl-c1">2</span>) <span class="pl-k">-</span> xi0) <span class="pl-k">*</span> log(<span class="pl-c1">1</span><span class="pl-k">-</span>xi3) <span class="pl-c"><span class="pl-c">#</span> t1 is HUGE, and because of numerical discrepancy, even though</span></td>
      </tr>
      <tr>
        <td id="L1503" class="blob-num js-line-number" data-line-number="1503"></td>
        <td id="LC1503" class="blob-code blob-code-inner js-file-line">                                                                <span class="pl-c"><span class="pl-c">#</span> if there is only 1 component, t1 approx 0,</span></td>
      </tr>
      <tr>
        <td id="L1504" class="blob-num js-line-number" data-line-number="1504"></td>
        <td id="LC1504" class="blob-code blob-code-inner js-file-line">                                                                <span class="pl-c"><span class="pl-c">#</span> but because each term is ~ 10^27 so discrepancies could be millions even</span></td>
      </tr>
      <tr>
        <td id="L1505" class="blob-num js-line-number" data-line-number="1505"></td>
        <td id="LC1505" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> xi1 <span class="pl-k">*</span> xi2 <span class="pl-k">/</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> xi3)</td>
      </tr>
      <tr>
        <td id="L1506" class="blob-num js-line-number" data-line-number="1506"></td>
        <td id="LC1506" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(xi2, <span class="pl-c1">3</span>) <span class="pl-k">/</span> (xi3 <span class="pl-k">*</span> <span class="pl-c1">pow</span>((<span class="pl-c1">1</span><span class="pl-k">-</span>xi3),<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L1507" class="blob-num js-line-number" data-line-number="1507"></td>
        <td id="LC1507" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> print(&#39;t2 + t3 gives&#39;, t2+t3)</span></td>
      </tr>
      <tr>
        <td id="L1508" class="blob-num js-line-number" data-line-number="1508"></td>
        <td id="LC1508" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> print(&#39;With pure imp:&#39;,(4*xi3 - 3*pow(xi3,2))/pow(1-xi3,2))</span></td>
      </tr>
      <tr>
        <td id="L1509" class="blob-num js-line-number" data-line-number="1509"></td>
        <td id="LC1509" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1510" class="blob-num js-line-number" data-line-number="1510"></td>
        <td id="LC1510" class="blob-code blob-code-inner js-file-line">        xiterm <span class="pl-k">=</span> t1 <span class="pl-k">+</span> t2 <span class="pl-k">+</span> t3 <span class="pl-c"><span class="pl-c">#</span> nm-3</span></td>
      </tr>
      <tr>
        <td id="L1511" class="blob-num js-line-number" data-line-number="1511"></td>
        <td id="LC1511" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">6</span> <span class="pl-k">*</span> xiterm <span class="pl-k">/</span> (pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.n_den()) <span class="pl-c"><span class="pl-c">#</span> uses n_den instead of segden</span></td>
      </tr>
      <tr>
        <td id="L1512" class="blob-num js-line-number" data-line-number="1512"></td>
        <td id="LC1512" class="blob-code blob-code-inner js-file-line">                                                  <span class="pl-c"><span class="pl-c">#</span> because it&#39;s just gonna multiply segratio again</span></td>
      </tr>
      <tr>
        <td id="L1513" class="blob-num js-line-number" data-line-number="1513"></td>
        <td id="LC1513" class="blob-code blob-code-inner js-file-line">                                                  <span class="pl-c"><span class="pl-c">#</span> so cancel it out</span></td>
      </tr>
      <tr>
        <td id="L1514" class="blob-num js-line-number" data-line-number="1514"></td>
        <td id="LC1514" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1515" class="blob-num js-line-number" data-line-number="1515"></td>
        <td id="LC1515" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1516" class="blob-num js-line-number" data-line-number="1516"></td>
        <td id="LC1516" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_1</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1517" class="blob-num js-line-number" data-line-number="1517"></td>
        <td id="LC1517" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1518" class="blob-num js-line-number" data-line-number="1518"></td>
        <td id="LC1518" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1/kT * cgss * sum of sum of _g1/cgss _g2/cgss * a1kl</span></td>
      </tr>
      <tr>
        <td id="L1519" class="blob-num js-line-number" data-line-number="1519"></td>
        <td id="LC1519" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1520" class="blob-num js-line-number" data-line-number="1520"></td>
        <td id="LC1520" class="blob-code blob-code-inner js-file-line">        a1sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1521" class="blob-num js-line-number" data-line-number="1521"></td>
        <td id="LC1521" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1522" class="blob-num js-line-number" data-line-number="1522"></td>
        <td id="LC1522" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1523" class="blob-num js-line-number" data-line-number="1523"></td>
        <td id="LC1523" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1524" class="blob-num js-line-number" data-line-number="1524"></td>
        <td id="LC1524" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf  <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1525" class="blob-num js-line-number" data-line-number="1525"></td>
        <td id="LC1525" class="blob-code blob-code-inner js-file-line">                a1kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_1kl(g1.index, g2.index)</td>
      </tr>
      <tr>
        <td id="L1526" class="blob-num js-line-number" data-line-number="1526"></td>
        <td id="LC1526" class="blob-code blob-code-inner js-file-line">                a1sum <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> a1kl</td>
      </tr>
      <tr>
        <td id="L1527" class="blob-num js-line-number" data-line-number="1527"></td>
        <td id="LC1527" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> (cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature) <span class="pl-k">*</span> a1sum </td>
      </tr>
      <tr>
        <td id="L1528" class="blob-num js-line-number" data-line-number="1528"></td>
        <td id="LC1528" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1529" class="blob-num js-line-number" data-line-number="1529"></td>
        <td id="LC1529" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1530" class="blob-num js-line-number" data-line-number="1530"></td>
        <td id="LC1530" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">g1i</span>, <span class="pl-smi">g2i</span>):</td>
      </tr>
      <tr>
        <td id="L1531" class="blob-num js-line-number" data-line-number="1531"></td>
        <td id="LC1531" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1532" class="blob-num js-line-number" data-line-number="1532"></td>
        <td id="LC1532" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Ckl * [ x0kl^att(as1kl(att) + bkl(att)) - x0kl^rep(as1kl(rep) + bkl(rep)) ]</span></td>
      </tr>
      <tr>
        <td id="L1533" class="blob-num js-line-number" data-line-number="1533"></td>
        <td id="LC1533" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1534" class="blob-num js-line-number" data-line-number="1534"></td>
        <td id="LC1534" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1535" class="blob-num js-line-number" data-line-number="1535"></td>
        <td id="LC1535" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.ghsd[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1536" class="blob-num js-line-number" data-line-number="1536"></td>
        <td id="LC1536" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1537" class="blob-num js-line-number" data-line-number="1537"></td>
        <td id="LC1537" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1538" class="blob-num js-line-number" data-line-number="1538"></td>
        <td id="LC1538" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> comb.rep</td>
      </tr>
      <tr>
        <td id="L1539" class="blob-num js-line-number" data-line-number="1539"></td>
        <td id="LC1539" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> comb.att</td>
      </tr>
      <tr>
        <td id="L1540" class="blob-num js-line-number" data-line-number="1540"></td>
        <td id="LC1540" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> comb.premie()</td>
      </tr>
      <tr>
        <td id="L1541" class="blob-num js-line-number" data-line-number="1541"></td>
        <td id="LC1541" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1542" class="blob-num js-line-number" data-line-number="1542"></td>
        <td id="LC1542" class="blob-code blob-code-inner js-file-line">        as1kl_a <span class="pl-k">=</span> <span class="pl-c1">self</span>.__as1kl(comb, hsd, att)</td>
      </tr>
      <tr>
        <td id="L1543" class="blob-num js-line-number" data-line-number="1543"></td>
        <td id="LC1543" class="blob-code blob-code-inner js-file-line">        bkl_a <span class="pl-k">=</span> <span class="pl-c1">self</span>.__bkl(comb, hsd, att)</td>
      </tr>
      <tr>
        <td id="L1544" class="blob-num js-line-number" data-line-number="1544"></td>
        <td id="LC1544" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1545" class="blob-num js-line-number" data-line-number="1545"></td>
        <td id="LC1545" class="blob-code blob-code-inner js-file-line">        as1kl_r <span class="pl-k">=</span> <span class="pl-c1">self</span>.__as1kl(comb, hsd, rep)</td>
      </tr>
      <tr>
        <td id="L1546" class="blob-num js-line-number" data-line-number="1546"></td>
        <td id="LC1546" class="blob-code blob-code-inner js-file-line">        bkl_r <span class="pl-k">=</span> <span class="pl-c1">self</span>.__bkl(comb, hsd, rep)</td>
      </tr>
      <tr>
        <td id="L1547" class="blob-num js-line-number" data-line-number="1547"></td>
        <td id="LC1547" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1548" class="blob-num js-line-number" data-line-number="1548"></td>
        <td id="LC1548" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, att) <span class="pl-k">*</span> (as1kl_a <span class="pl-k">+</span> bkl_a)</td>
      </tr>
      <tr>
        <td id="L1549" class="blob-num js-line-number" data-line-number="1549"></td>
        <td id="LC1549" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, rep) <span class="pl-k">*</span> (as1kl_r <span class="pl-k">+</span> bkl_r)</td>
      </tr>
      <tr>
        <td id="L1550" class="blob-num js-line-number" data-line-number="1550"></td>
        <td id="LC1550" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> ckl <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2)</td>
      </tr>
      <tr>
        <td id="L1551" class="blob-num js-line-number" data-line-number="1551"></td>
        <td id="LC1551" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1552" class="blob-num js-line-number" data-line-number="1552"></td>
        <td id="LC1552" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1553" class="blob-num js-line-number" data-line-number="1553"></td>
        <td id="LC1553" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1554" class="blob-num js-line-number" data-line-number="1554"></td>
        <td id="LC1554" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1555" class="blob-num js-line-number" data-line-number="1555"></td>
        <td id="LC1555" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1556" class="blob-num js-line-number" data-line-number="1556"></td>
        <td id="LC1556" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1/kT)**2 * cgss * sum of sum of _g1/cgss _g2/cgss * a2kl</span></td>
      </tr>
      <tr>
        <td id="L1557" class="blob-num js-line-number" data-line-number="1557"></td>
        <td id="LC1557" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1558" class="blob-num js-line-number" data-line-number="1558"></td>
        <td id="LC1558" class="blob-code blob-code-inner js-file-line">        a2sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1559" class="blob-num js-line-number" data-line-number="1559"></td>
        <td id="LC1559" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1560" class="blob-num js-line-number" data-line-number="1560"></td>
        <td id="LC1560" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1561" class="blob-num js-line-number" data-line-number="1561"></td>
        <td id="LC1561" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1562" class="blob-num js-line-number" data-line-number="1562"></td>
        <td id="LC1562" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1563" class="blob-num js-line-number" data-line-number="1563"></td>
        <td id="LC1563" class="blob-code blob-code-inner js-file-line">                a2kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_2kl(g1.index, g2.index)</td>
      </tr>
      <tr>
        <td id="L1564" class="blob-num js-line-number" data-line-number="1564"></td>
        <td id="LC1564" class="blob-code blob-code-inner js-file-line">                a2sum <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> a2kl </td>
      </tr>
      <tr>
        <td id="L1565" class="blob-num js-line-number" data-line-number="1565"></td>
        <td id="LC1565" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> <span class="pl-c1">pow</span>((cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature),<span class="pl-c1">2</span>) <span class="pl-k">*</span> a2sum </td>
      </tr>
      <tr>
        <td id="L1566" class="blob-num js-line-number" data-line-number="1566"></td>
        <td id="LC1566" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1567" class="blob-num js-line-number" data-line-number="1567"></td>
        <td id="LC1567" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1568" class="blob-num js-line-number" data-line-number="1568"></td>
        <td id="LC1568" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_2kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">g1i</span>, <span class="pl-smi">g2i</span>):</td>
      </tr>
      <tr>
        <td id="L1569" class="blob-num js-line-number" data-line-number="1569"></td>
        <td id="LC1569" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1570" class="blob-num js-line-number" data-line-number="1570"></td>
        <td id="LC1570" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1/2 * khs * (1-corrf) * epsikl * ckl^2 *</span></td>
      </tr>
      <tr>
        <td id="L1571" class="blob-num js-line-number" data-line-number="1571"></td>
        <td id="LC1571" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        { x0kl^(2att) * (as1kl(2*att) + bkl(2*att))</span></td>
      </tr>
      <tr>
        <td id="L1572" class="blob-num js-line-number" data-line-number="1572"></td>
        <td id="LC1572" class="blob-code blob-code-inner js-file-line"><span class="pl-s">          - 2*x0kl^(att+rep) * (as1kl(att+rep) + bkl(att+rep))</span></td>
      </tr>
      <tr>
        <td id="L1573" class="blob-num js-line-number" data-line-number="1573"></td>
        <td id="LC1573" class="blob-code blob-code-inner js-file-line"><span class="pl-s">          + x0kl^(2rep) * (as1kl(2rep) + bkl(2rep)) }</span></td>
      </tr>
      <tr>
        <td id="L1574" class="blob-num js-line-number" data-line-number="1574"></td>
        <td id="LC1574" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1575" class="blob-num js-line-number" data-line-number="1575"></td>
        <td id="LC1575" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1576" class="blob-num js-line-number" data-line-number="1576"></td>
        <td id="LC1576" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.ghsd[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1577" class="blob-num js-line-number" data-line-number="1577"></td>
        <td id="LC1577" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L1578" class="blob-num js-line-number" data-line-number="1578"></td>
        <td id="LC1578" class="blob-code blob-code-inner js-file-line">        corrf <span class="pl-k">=</span> <span class="pl-c1">self</span>.__corrf(comb)</td>
      </tr>
      <tr>
        <td id="L1579" class="blob-num js-line-number" data-line-number="1579"></td>
        <td id="LC1579" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1580" class="blob-num js-line-number" data-line-number="1580"></td>
        <td id="LC1580" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> comb.premie()</td>
      </tr>
      <tr>
        <td id="L1581" class="blob-num js-line-number" data-line-number="1581"></td>
        <td id="LC1581" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> comb.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1582" class="blob-num js-line-number" data-line-number="1582"></td>
        <td id="LC1582" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> comb.rep</td>
      </tr>
      <tr>
        <td id="L1583" class="blob-num js-line-number" data-line-number="1583"></td>
        <td id="LC1583" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> comb.att</td>
      </tr>
      <tr>
        <td id="L1584" class="blob-num js-line-number" data-line-number="1584"></td>
        <td id="LC1584" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1585" class="blob-num js-line-number" data-line-number="1585"></td>
        <td id="LC1585" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(comb, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(comb, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L1586" class="blob-num js-line-number" data-line-number="1586"></td>
        <td id="LC1586" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(comb, hsd, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(comb, hsd, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L1587" class="blob-num js-line-number" data-line-number="1587"></td>
        <td id="LC1587" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(comb, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(comb, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L1588" class="blob-num js-line-number" data-line-number="1588"></td>
        <td id="LC1588" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1589" class="blob-num js-line-number" data-line-number="1589"></td>
        <td id="LC1589" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">0.5</span> <span class="pl-k">*</span> khs <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">+</span>corrf) <span class="pl-k">*</span> epsikl <span class="pl-k">*</span> <span class="pl-c1">pow</span>(ckl, <span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1590" class="blob-num js-line-number" data-line-number="1590"></td>
        <td id="LC1590" class="blob-code blob-code-inner js-file-line">        res <span class="pl-k">=</span> t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3</td>
      </tr>
      <tr>
        <td id="L1591" class="blob-num js-line-number" data-line-number="1591"></td>
        <td id="LC1591" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1592" class="blob-num js-line-number" data-line-number="1592"></td>
        <td id="LC1592" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> fac <span class="pl-k">*</span> res</td>
      </tr>
      <tr>
        <td id="L1593" class="blob-num js-line-number" data-line-number="1593"></td>
        <td id="LC1593" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1594" class="blob-num js-line-number" data-line-number="1594"></td>
        <td id="LC1594" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1595" class="blob-num js-line-number" data-line-number="1595"></td>
        <td id="LC1595" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__khs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1596" class="blob-num js-line-number" data-line-number="1596"></td>
        <td id="LC1596" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1597" class="blob-num js-line-number" data-line-number="1597"></td>
        <td id="LC1597" class="blob-code blob-code-inner js-file-line">        num <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix, <span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L1598" class="blob-num js-line-number" data-line-number="1598"></td>
        <td id="LC1598" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span>xix <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">-</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>) <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L1599" class="blob-num js-line-number" data-line-number="1599"></td>
        <td id="LC1599" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> num<span class="pl-k">/</span>den</td>
      </tr>
      <tr>
        <td id="L1600" class="blob-num js-line-number" data-line-number="1600"></td>
        <td id="LC1600" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1601" class="blob-num js-line-number" data-line-number="1601"></td>
        <td id="LC1601" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__corrf</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>):</td>
      </tr>
      <tr>
        <td id="L1602" class="blob-num js-line-number" data-line-number="1602"></td>
        <td id="LC1602" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L1603" class="blob-num js-line-number" data-line-number="1603"></td>
        <td id="LC1603" class="blob-code blob-code-inner js-file-line">        alkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(comb)</td>
      </tr>
      <tr>
        <td id="L1604" class="blob-num js-line-number" data-line-number="1604"></td>
        <td id="LC1604" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1605" class="blob-num js-line-number" data-line-number="1605"></td>
        <td id="LC1605" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">1</span>) <span class="pl-k">*</span> xisx</td>
      </tr>
      <tr>
        <td id="L1606" class="blob-num js-line-number" data-line-number="1606"></td>
        <td id="LC1606" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">2</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">5</span>)</td>
      </tr>
      <tr>
        <td id="L1607" class="blob-num js-line-number" data-line-number="1607"></td>
        <td id="LC1607" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">8</span>)</td>
      </tr>
      <tr>
        <td id="L1608" class="blob-num js-line-number" data-line-number="1608"></td>
        <td id="LC1608" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> t1 <span class="pl-k">+</span> t2 <span class="pl-k">+</span> t3</td>
      </tr>
      <tr>
        <td id="L1609" class="blob-num js-line-number" data-line-number="1609"></td>
        <td id="LC1609" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1610" class="blob-num js-line-number" data-line-number="1610"></td>
        <td id="LC1610" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1611" class="blob-num js-line-number" data-line-number="1611"></td>
        <td id="LC1611" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__alphakl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>):</td>
      </tr>
      <tr>
        <td id="L1612" class="blob-num js-line-number" data-line-number="1612"></td>
        <td id="LC1612" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (comb.att <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1613" class="blob-num js-line-number" data-line-number="1613"></td>
        <td id="LC1613" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (comb.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1614" class="blob-num js-line-number" data-line-number="1614"></td>
        <td id="LC1614" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> comb.premie() <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2) </td>
      </tr>
      <tr>
        <td id="L1615" class="blob-num js-line-number" data-line-number="1615"></td>
        <td id="LC1615" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1616" class="blob-num js-line-number" data-line-number="1616"></td>
        <td id="LC1616" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1617" class="blob-num js-line-number" data-line-number="1617"></td>
        <td id="LC1617" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1618" class="blob-num js-line-number" data-line-number="1618"></td>
        <td id="LC1618" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">a_3</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1619" class="blob-num js-line-number" data-line-number="1619"></td>
        <td id="LC1619" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1620" class="blob-num js-line-number" data-line-number="1620"></td>
        <td id="LC1620" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1/kT)**3 * cgss * sum of sum of _g1/cgss _g2/cgss * a3kl</span></td>
      </tr>
      <tr>
        <td id="L1621" class="blob-num js-line-number" data-line-number="1621"></td>
        <td id="LC1621" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1622" class="blob-num js-line-number" data-line-number="1622"></td>
        <td id="LC1622" class="blob-code blob-code-inner js-file-line">        a3sum <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1623" class="blob-num js-line-number" data-line-number="1623"></td>
        <td id="LC1623" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1624" class="blob-num js-line-number" data-line-number="1624"></td>
        <td id="LC1624" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1625" class="blob-num js-line-number" data-line-number="1625"></td>
        <td id="LC1625" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1626" class="blob-num js-line-number" data-line-number="1626"></td>
        <td id="LC1626" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1627" class="blob-num js-line-number" data-line-number="1627"></td>
        <td id="LC1627" class="blob-code blob-code-inner js-file-line">                a3kl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__a_3kl(g1.index, g2.index)</td>
      </tr>
      <tr>
        <td id="L1628" class="blob-num js-line-number" data-line-number="1628"></td>
        <td id="LC1628" class="blob-code blob-code-inner js-file-line">                a3sum <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> a3kl </td>
      </tr>
      <tr>
        <td id="L1629" class="blob-num js-line-number" data-line-number="1629"></td>
        <td id="LC1629" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">self</span>.segratio <span class="pl-k">/</span> <span class="pl-c1">pow</span>((cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature),<span class="pl-c1">3</span>) <span class="pl-k">*</span> a3sum </td>
      </tr>
      <tr>
        <td id="L1630" class="blob-num js-line-number" data-line-number="1630"></td>
        <td id="LC1630" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1631" class="blob-num js-line-number" data-line-number="1631"></td>
        <td id="LC1631" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1632" class="blob-num js-line-number" data-line-number="1632"></td>
        <td id="LC1632" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__a_3kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">g1i</span>, <span class="pl-smi">g2i</span>):</td>
      </tr>
      <tr>
        <td id="L1633" class="blob-num js-line-number" data-line-number="1633"></td>
        <td id="LC1633" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.gcomb[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1634" class="blob-num js-line-number" data-line-number="1634"></td>
        <td id="LC1634" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L1635" class="blob-num js-line-number" data-line-number="1635"></td>
        <td id="LC1635" class="blob-code blob-code-inner js-file-line">        alkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(comb)</td>
      </tr>
      <tr>
        <td id="L1636" class="blob-num js-line-number" data-line-number="1636"></td>
        <td id="LC1636" class="blob-code blob-code-inner js-file-line">        epsikl <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1637" class="blob-num js-line-number" data-line-number="1637"></td>
        <td id="LC1637" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1638" class="blob-num js-line-number" data-line-number="1638"></td>
        <td id="LC1638" class="blob-code blob-code-inner js-file-line">        preexp <span class="pl-k">=</span> <span class="pl-k">-</span> <span class="pl-c1">pow</span>(epsikl,<span class="pl-c1">3</span>) <span class="pl-k">*</span> mt.f_m(alkl, <span class="pl-c1">4</span>) <span class="pl-k">*</span> xisx</td>
      </tr>
      <tr>
        <td id="L1639" class="blob-num js-line-number" data-line-number="1639"></td>
        <td id="LC1639" class="blob-code blob-code-inner js-file-line">        expt1 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">5</span>) <span class="pl-k">*</span> xisx </td>
      </tr>
      <tr>
        <td id="L1640" class="blob-num js-line-number" data-line-number="1640"></td>
        <td id="LC1640" class="blob-code blob-code-inner js-file-line">        expt2 <span class="pl-k">=</span> mt.f_m(alkl, <span class="pl-c1">6</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1641" class="blob-num js-line-number" data-line-number="1641"></td>
        <td id="LC1641" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1642" class="blob-num js-line-number" data-line-number="1642"></td>
        <td id="LC1642" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> preexp <span class="pl-k">*</span> exp(expt1 <span class="pl-k">+</span> expt2)</td>
      </tr>
      <tr>
        <td id="L1643" class="blob-num js-line-number" data-line-number="1643"></td>
        <td id="LC1643" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1644" class="blob-num js-line-number" data-line-number="1644"></td>
        <td id="LC1644" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L1645" class="blob-num js-line-number" data-line-number="1645"></td>
        <td id="LC1645" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for chain</span></td>
      </tr>
      <tr>
        <td id="L1646" class="blob-num js-line-number" data-line-number="1646"></td>
        <td id="LC1646" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__gmieij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ci</span>):</td>
      </tr>
      <tr>
        <td id="L1647" class="blob-num js-line-number" data-line-number="1647"></td>
        <td id="LC1647" class="blob-code blob-code-inner js-file-line">        gdhs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__gdhs(ci)</td>
      </tr>
      <tr>
        <td id="L1648" class="blob-num js-line-number" data-line-number="1648"></td>
        <td id="LC1648" class="blob-code blob-code-inner js-file-line">        g1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g1(ci)</td>
      </tr>
      <tr>
        <td id="L1649" class="blob-num js-line-number" data-line-number="1649"></td>
        <td id="LC1649" class="blob-code blob-code-inner js-file-line">        g2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g2(ci)</td>
      </tr>
      <tr>
        <td id="L1650" class="blob-num js-line-number" data-line-number="1650"></td>
        <td id="LC1650" class="blob-code blob-code-inner js-file-line">        b <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (cst.k <span class="pl-k">*</span> <span class="pl-c1">self</span>.temperature)</td>
      </tr>
      <tr>
        <td id="L1651" class="blob-num js-line-number" data-line-number="1651"></td>
        <td id="LC1651" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.ccomb[ci]</td>
      </tr>
      <tr>
        <td id="L1652" class="blob-num js-line-number" data-line-number="1652"></td>
        <td id="LC1652" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1653" class="blob-num js-line-number" data-line-number="1653"></td>
        <td id="LC1653" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1654" class="blob-num js-line-number" data-line-number="1654"></td>
        <td id="LC1654" class="blob-code blob-code-inner js-file-line">        expt <span class="pl-k">=</span> b <span class="pl-k">*</span> epsi <span class="pl-k">*</span> g1 <span class="pl-k">/</span> gdhs <span class="pl-k">+</span> <span class="pl-c1">pow</span>(b<span class="pl-k">*</span>epsi,<span class="pl-c1">2</span>) <span class="pl-k">*</span> g2 <span class="pl-k">/</span> gdhs</td>
      </tr>
      <tr>
        <td id="L1655" class="blob-num js-line-number" data-line-number="1655"></td>
        <td id="LC1655" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> gdhs <span class="pl-k">*</span> exp(expt)</td>
      </tr>
      <tr>
        <td id="L1656" class="blob-num js-line-number" data-line-number="1656"></td>
        <td id="LC1656" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1657" class="blob-num js-line-number" data-line-number="1657"></td>
        <td id="LC1657" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1658" class="blob-num js-line-number" data-line-number="1658"></td>
        <td id="LC1658" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_xi_x</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1659" class="blob-num js-line-number" data-line-number="1659"></td>
        <td id="LC1659" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> pi <span class="pl-k">/</span> <span class="pl-c1">6</span></td>
      </tr>
      <tr>
        <td id="L1660" class="blob-num js-line-number" data-line-number="1660"></td>
        <td id="LC1660" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L1661" class="blob-num js-line-number" data-line-number="1661"></td>
        <td id="LC1661" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1662" class="blob-num js-line-number" data-line-number="1662"></td>
        <td id="LC1662" class="blob-code blob-code-inner js-file-line">            g1i <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L1663" class="blob-num js-line-number" data-line-number="1663"></td>
        <td id="LC1663" class="blob-code blob-code-inner js-file-line">            xsk <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1664" class="blob-num js-line-number" data-line-number="1664"></td>
        <td id="LC1664" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L1665" class="blob-num js-line-number" data-line-number="1665"></td>
        <td id="LC1665" class="blob-code blob-code-inner js-file-line">                g2i <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1666" class="blob-num js-line-number" data-line-number="1666"></td>
        <td id="LC1666" class="blob-code blob-code-inner js-file-line">                xsl <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-k">/</span> <span class="pl-c1">self</span>.segratio</td>
      </tr>
      <tr>
        <td id="L1667" class="blob-num js-line-number" data-line-number="1667"></td>
        <td id="LC1667" class="blob-code blob-code-inner js-file-line">                dkl <span class="pl-k">=</span> <span class="pl-c1">self</span>.ghsd[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L1668" class="blob-num js-line-number" data-line-number="1668"></td>
        <td id="LC1668" class="blob-code blob-code-inner js-file-line">                result <span class="pl-k">+=</span> xsk <span class="pl-k">*</span> xsl <span class="pl-k">*</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1669" class="blob-num js-line-number" data-line-number="1669"></td>
        <td id="LC1669" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> fac <span class="pl-k">*</span> result</td>
      </tr>
      <tr>
        <td id="L1670" class="blob-num js-line-number" data-line-number="1670"></td>
        <td id="LC1670" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1671" class="blob-num js-line-number" data-line-number="1671"></td>
        <td id="LC1671" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_as1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>, <span class="pl-smi">dkl</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1672" class="blob-num js-line-number" data-line-number="1672"></td>
        <td id="LC1672" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1673" class="blob-num js-line-number" data-line-number="1673"></td>
        <td id="LC1673" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        -2 * pi * (ep*d^3/(lam-3)) * (1/(1-xieff)^3) * (1 - xieff/2 + segden * derxieff * (3*(1-xieff/2)/(1-xieff) - 1/2))</span></td>
      </tr>
      <tr>
        <td id="L1674" class="blob-num js-line-number" data-line-number="1674"></td>
        <td id="LC1674" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1675" class="blob-num js-line-number" data-line-number="1675"></td>
        <td id="LC1675" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1676" class="blob-num js-line-number" data-line-number="1676"></td>
        <td id="LC1676" class="blob-code blob-code-inner js-file-line">        d_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1677" class="blob-num js-line-number" data-line-number="1677"></td>
        <td id="LC1677" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1678" class="blob-num js-line-number" data-line-number="1678"></td>
        <td id="LC1678" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1679" class="blob-num js-line-number" data-line-number="1679"></td>
        <td id="LC1679" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1680" class="blob-num js-line-number" data-line-number="1680"></td>
        <td id="LC1680" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1681" class="blob-num js-line-number" data-line-number="1681"></td>
        <td id="LC1681" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1682" class="blob-num js-line-number" data-line-number="1682"></td>
        <td id="LC1682" class="blob-code blob-code-inner js-file-line">        derxixeff <span class="pl-k">=</span> mt.der_xi_x_eff(xix, derxix, lam)</td>
      </tr>
      <tr>
        <td id="L1683" class="blob-num js-line-number" data-line-number="1683"></td>
        <td id="LC1683" class="blob-code blob-code-inner js-file-line">        xixeff <span class="pl-k">=</span> mt.xi_x_eff(xix, lam)</td>
      </tr>
      <tr>
        <td id="L1684" class="blob-num js-line-number" data-line-number="1684"></td>
        <td id="LC1684" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1685" class="blob-num js-line-number" data-line-number="1685"></td>
        <td id="LC1685" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> d_3 <span class="pl-k">/</span> (lam<span class="pl-k">-</span><span class="pl-c1">3</span>) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1686" class="blob-num js-line-number" data-line-number="1686"></td>
        <td id="LC1686" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xixeff <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1687" class="blob-num js-line-number" data-line-number="1687"></td>
        <td id="LC1687" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> segden <span class="pl-k">*</span> derxixeff</td>
      </tr>
      <tr>
        <td id="L1688" class="blob-num js-line-number" data-line-number="1688"></td>
        <td id="LC1688" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff<span class="pl-k">/</span><span class="pl-c1">2</span>) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xixeff)</td>
      </tr>
      <tr>
        <td id="L1689" class="blob-num js-line-number" data-line-number="1689"></td>
        <td id="LC1689" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1690" class="blob-num js-line-number" data-line-number="1690"></td>
        <td id="LC1690" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-k">-</span> fac <span class="pl-k">*</span> (t1 <span class="pl-k">+</span> t2 <span class="pl-k">*</span> (t3 <span class="pl-k">-</span> <span class="pl-c1">1</span><span class="pl-k">/</span><span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L1691" class="blob-num js-line-number" data-line-number="1691"></td>
        <td id="LC1691" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1692" class="blob-num js-line-number" data-line-number="1692"></td>
        <td id="LC1692" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1693" class="blob-num js-line-number" data-line-number="1693"></td>
        <td id="LC1693" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1694" class="blob-num js-line-number" data-line-number="1694"></td>
        <td id="LC1694" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_bkl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">comb</span>, <span class="pl-smi">dkl</span>, <span class="pl-smi">lam</span>):</td>
      </tr>
      <tr>
        <td id="L1695" class="blob-num js-line-number" data-line-number="1695"></td>
        <td id="LC1695" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1696" class="blob-num js-line-number" data-line-number="1696"></td>
        <td id="LC1696" class="blob-code blob-code-inner js-file-line">        d_3 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(dkl, <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1697" class="blob-num js-line-number" data-line-number="1697"></td>
        <td id="LC1697" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> comb.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1698" class="blob-num js-line-number" data-line-number="1698"></td>
        <td id="LC1698" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1699" class="blob-num js-line-number" data-line-number="1699"></td>
        <td id="LC1699" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> comb.x0kl(dkl)</td>
      </tr>
      <tr>
        <td id="L1700" class="blob-num js-line-number" data-line-number="1700"></td>
        <td id="LC1700" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1701" class="blob-num js-line-number" data-line-number="1701"></td>
        <td id="LC1701" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1702" class="blob-num js-line-number" data-line-number="1702"></td>
        <td id="LC1702" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1703" class="blob-num js-line-number" data-line-number="1703"></td>
        <td id="LC1703" class="blob-code blob-code-inner js-file-line">        fac <span class="pl-k">=</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> d_3 <span class="pl-k">/</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1704" class="blob-num js-line-number" data-line-number="1704"></td>
        <td id="LC1704" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1705" class="blob-num js-line-number" data-line-number="1705"></td>
        <td id="LC1705" class="blob-code blob-code-inner js-file-line">        t11 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> xix<span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1706" class="blob-num js-line-number" data-line-number="1706"></td>
        <td id="LC1706" class="blob-code blob-code-inner js-file-line">        t12 <span class="pl-k">=</span> segden <span class="pl-k">*</span> derxix</td>
      </tr>
      <tr>
        <td id="L1707" class="blob-num js-line-number" data-line-number="1707"></td>
        <td id="LC1707" class="blob-code blob-code-inner js-file-line">        t13 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix<span class="pl-k">/</span><span class="pl-c1">2</span>) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix) <span class="pl-k">-</span> <span class="pl-c1">1</span><span class="pl-k">/</span><span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1708" class="blob-num js-line-number" data-line-number="1708"></td>
        <td id="LC1708" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> t11 <span class="pl-k">+</span> t12 <span class="pl-k">*</span> t13</td>
      </tr>
      <tr>
        <td id="L1709" class="blob-num js-line-number" data-line-number="1709"></td>
        <td id="LC1709" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1710" class="blob-num js-line-number" data-line-number="1710"></td>
        <td id="LC1710" class="blob-code blob-code-inner js-file-line">        t21 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1711" class="blob-num js-line-number" data-line-number="1711"></td>
        <td id="LC1711" class="blob-code blob-code-inner js-file-line">        t22 <span class="pl-k">=</span> segden<span class="pl-k">*</span>derxix</td>
      </tr>
      <tr>
        <td id="L1712" class="blob-num js-line-number" data-line-number="1712"></td>
        <td id="LC1712" class="blob-code blob-code-inner js-file-line">        t23 <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">2</span> <span class="pl-k">*</span> xix <span class="pl-k">+</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> xix <span class="pl-k">*</span> (<span class="pl-c1">1</span><span class="pl-k">+</span>xix) <span class="pl-k">/</span> (<span class="pl-c1">1</span><span class="pl-k">-</span>xix) </td>
      </tr>
      <tr>
        <td id="L1713" class="blob-num js-line-number" data-line-number="1713"></td>
        <td id="LC1713" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">9</span><span class="pl-k">/</span><span class="pl-c1">2</span> <span class="pl-k">*</span> (t21 <span class="pl-k">+</span> t22 <span class="pl-k">*</span> t23)</td>
      </tr>
      <tr>
        <td id="L1714" class="blob-num js-line-number" data-line-number="1714"></td>
        <td id="LC1714" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1715" class="blob-num js-line-number" data-line-number="1715"></td>
        <td id="LC1715" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> fac <span class="pl-k">*</span> (t1 <span class="pl-k">*</span> mt.intI(x0ii,lam) <span class="pl-k">-</span> t2 <span class="pl-k">*</span> mt.intJ(x0ii,lam))</td>
      </tr>
      <tr>
        <td id="L1716" class="blob-num js-line-number" data-line-number="1716"></td>
        <td id="LC1716" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1717" class="blob-num js-line-number" data-line-number="1717"></td>
        <td id="LC1717" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1718" class="blob-num js-line-number" data-line-number="1718"></td>
        <td id="LC1718" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_a1kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">gcomp</span>, <span class="pl-smi">hsdii</span>):</td>
      </tr>
      <tr>
        <td id="L1719" class="blob-num js-line-number" data-line-number="1719"></td>
        <td id="LC1719" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(hsdii)</td>
      </tr>
      <tr>
        <td id="L1720" class="blob-num js-line-number" data-line-number="1720"></td>
        <td id="LC1720" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1721" class="blob-num js-line-number" data-line-number="1721"></td>
        <td id="LC1721" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1722" class="blob-num js-line-number" data-line-number="1722"></td>
        <td id="LC1722" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1723" class="blob-num js-line-number" data-line-number="1723"></td>
        <td id="LC1723" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1724" class="blob-num js-line-number" data-line-number="1724"></td>
        <td id="LC1724" class="blob-code blob-code-inner js-file-line">        tatt <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_as1kl(gcomp, hsdii, att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(gcomp, hsdii, att)</td>
      </tr>
      <tr>
        <td id="L1725" class="blob-num js-line-number" data-line-number="1725"></td>
        <td id="LC1725" class="blob-code blob-code-inner js-file-line">        trep <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_as1kl(gcomp, hsdii, rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(gcomp, hsdii, rep)</td>
      </tr>
      <tr>
        <td id="L1726" class="blob-num js-line-number" data-line-number="1726"></td>
        <td id="LC1726" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1727" class="blob-num js-line-number" data-line-number="1727"></td>
        <td id="LC1727" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> ckl <span class="pl-k">*</span> (<span class="pl-c1">pow</span>(x0ii, att) <span class="pl-k">*</span> tatt <span class="pl-k">-</span> <span class="pl-c1">pow</span>(x0ii, rep) <span class="pl-k">*</span> trep)</td>
      </tr>
      <tr>
        <td id="L1728" class="blob-num js-line-number" data-line-number="1728"></td>
        <td id="LC1728" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1729" class="blob-num js-line-number" data-line-number="1729"></td>
        <td id="LC1729" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1730" class="blob-num js-line-number" data-line-number="1730"></td>
        <td id="LC1730" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__gdhs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ci</span>):</td>
      </tr>
      <tr>
        <td id="L1731" class="blob-num js-line-number" data-line-number="1731"></td>
        <td id="LC1731" class="blob-code blob-code-inner js-file-line">        xi_x <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1732" class="blob-num js-line-number" data-line-number="1732"></td>
        <td id="LC1732" class="blob-code blob-code-inner js-file-line">        comb <span class="pl-k">=</span> <span class="pl-c1">self</span>.ccomb[ci]</td>
      </tr>
      <tr>
        <td id="L1733" class="blob-num js-line-number" data-line-number="1733"></td>
        <td id="LC1733" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> comb.x0kl(<span class="pl-c1">self</span>.chsd[ci])</td>
      </tr>
      <tr>
        <td id="L1734" class="blob-num js-line-number" data-line-number="1734"></td>
        <td id="LC1734" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1735" class="blob-num js-line-number" data-line-number="1735"></td>
        <td id="LC1735" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> mt.gdhs(x0ii, xi_x)</td>
      </tr>
      <tr>
        <td id="L1736" class="blob-num js-line-number" data-line-number="1736"></td>
        <td id="LC1736" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1737" class="blob-num js-line-number" data-line-number="1737"></td>
        <td id="LC1737" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1738" class="blob-num js-line-number" data-line-number="1738"></td>
        <td id="LC1738" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g1</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ci</span>):</td>
      </tr>
      <tr>
        <td id="L1739" class="blob-num js-line-number" data-line-number="1739"></td>
        <td id="LC1739" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1740" class="blob-num js-line-number" data-line-number="1740"></td>
        <td id="LC1740" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1 / (2pi epsiii hsdii^3) * [ 3 da1kl/dp - premieii * attii * x0ii^attii * (as1kl(attii) + Bkl(attii))/segden </span></td>
      </tr>
      <tr>
        <td id="L1741" class="blob-num js-line-number" data-line-number="1741"></td>
        <td id="LC1741" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                                     +  premieii * repii * x0ii^repii * (as1kl(repii) + Bkl(repii))/segden ]</span></td>
      </tr>
      <tr>
        <td id="L1742" class="blob-num js-line-number" data-line-number="1742"></td>
        <td id="LC1742" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1743" class="blob-num js-line-number" data-line-number="1743"></td>
        <td id="LC1743" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.ccomb[ci]</td>
      </tr>
      <tr>
        <td id="L1744" class="blob-num js-line-number" data-line-number="1744"></td>
        <td id="LC1744" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.chsd[ci]</td>
      </tr>
      <tr>
        <td id="L1745" class="blob-num js-line-number" data-line-number="1745"></td>
        <td id="LC1745" class="blob-code blob-code-inner js-file-line">        premie <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1746" class="blob-num js-line-number" data-line-number="1746"></td>
        <td id="LC1746" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1747" class="blob-num js-line-number" data-line-number="1747"></td>
        <td id="LC1747" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1748" class="blob-num js-line-number" data-line-number="1748"></td>
        <td id="LC1748" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1749" class="blob-num js-line-number" data-line-number="1749"></td>
        <td id="LC1749" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1750" class="blob-num js-line-number" data-line-number="1750"></td>
        <td id="LC1750" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1751" class="blob-num js-line-number" data-line-number="1751"></td>
        <td id="LC1751" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1752" class="blob-num js-line-number" data-line-number="1752"></td>
        <td id="LC1752" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1753" class="blob-num js-line-number" data-line-number="1753"></td>
        <td id="LC1753" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">3</span><span class="pl-k">*</span> <span class="pl-c1">self</span>.__der_a1kl(gcomp, hsd)</td>
      </tr>
      <tr>
        <td id="L1754" class="blob-num js-line-number" data-line-number="1754"></td>
        <td id="LC1754" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> premie <span class="pl-k">*</span> att <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd, att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd, att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1755" class="blob-num js-line-number" data-line-number="1755"></td>
        <td id="LC1755" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> premie <span class="pl-k">*</span> rep <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd, rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd, rep)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1756" class="blob-num js-line-number" data-line-number="1756"></td>
        <td id="LC1756" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1757" class="blob-num js-line-number" data-line-number="1757"></td>
        <td id="LC1757" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (<span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> epsi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(hsd,<span class="pl-c1">3</span>)) <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3)</td>
      </tr>
      <tr>
        <td id="L1758" class="blob-num js-line-number" data-line-number="1758"></td>
        <td id="LC1758" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1759" class="blob-num js-line-number" data-line-number="1759"></td>
        <td id="LC1759" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1760" class="blob-num js-line-number" data-line-number="1760"></td>
        <td id="LC1760" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1761" class="blob-num js-line-number" data-line-number="1761"></td>
        <td id="LC1761" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g2</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ci</span>):</td>
      </tr>
      <tr>
        <td id="L1762" class="blob-num js-line-number" data-line-number="1762"></td>
        <td id="LC1762" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1763" class="blob-num js-line-number" data-line-number="1763"></td>
        <td id="LC1763" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (1+gammacii) * g2MCA(hsdii)</span></td>
      </tr>
      <tr>
        <td id="L1764" class="blob-num js-line-number" data-line-number="1764"></td>
        <td id="LC1764" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1765" class="blob-num js-line-number" data-line-number="1765"></td>
        <td id="LC1765" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.ccomb[ci]</td>
      </tr>
      <tr>
        <td id="L1766" class="blob-num js-line-number" data-line-number="1766"></td>
        <td id="LC1766" class="blob-code blob-code-inner js-file-line">        xisx <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_sx()</td>
      </tr>
      <tr>
        <td id="L1767" class="blob-num js-line-number" data-line-number="1767"></td>
        <td id="LC1767" class="blob-code blob-code-inner js-file-line">        alii <span class="pl-k">=</span> <span class="pl-c1">self</span>.__alphakl(gcomp)</td>
      </tr>
      <tr>
        <td id="L1768" class="blob-num js-line-number" data-line-number="1768"></td>
        <td id="LC1768" class="blob-code blob-code-inner js-file-line">        theta <span class="pl-k">=</span> exp(gcomp.epsilon <span class="pl-k">/</span> <span class="pl-c1">self</span>.temperature) <span class="pl-k">-</span><span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L1769" class="blob-num js-line-number" data-line-number="1769"></td>
        <td id="LC1769" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1770" class="blob-num js-line-number" data-line-number="1770"></td>
        <td id="LC1770" class="blob-code blob-code-inner js-file-line">        gammacii <span class="pl-k">=</span> cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">0</span>] <span class="pl-k">*</span> (<span class="pl-k">-</span>tanh(cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">1</span>] <span class="pl-k">*</span> (cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">2</span>]<span class="pl-k">-</span>alii)) <span class="pl-k">+</span> <span class="pl-c1">1</span>) <span class="pl-k">*</span> xisx <span class="pl-k">*</span> theta <span class="pl-k">*</span> exp(cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">3</span>]<span class="pl-k">*</span>xisx <span class="pl-k">+</span> cst.phi[<span class="pl-c1">6</span>,<span class="pl-c1">4</span>] <span class="pl-k">*</span> <span class="pl-c1">pow</span>(xisx, <span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L1771" class="blob-num js-line-number" data-line-number="1771"></td>
        <td id="LC1771" class="blob-code blob-code-inner js-file-line">        g2mca <span class="pl-k">=</span> <span class="pl-c1">self</span>.__g2mca(ci)</td>
      </tr>
      <tr>
        <td id="L1772" class="blob-num js-line-number" data-line-number="1772"></td>
        <td id="LC1772" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> (<span class="pl-c1">1</span> <span class="pl-k">+</span> gammacii) <span class="pl-k">*</span> g2mca</td>
      </tr>
      <tr>
        <td id="L1773" class="blob-num js-line-number" data-line-number="1773"></td>
        <td id="LC1773" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1774" class="blob-num js-line-number" data-line-number="1774"></td>
        <td id="LC1774" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1775" class="blob-num js-line-number" data-line-number="1775"></td>
        <td id="LC1775" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_khs</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1776" class="blob-num js-line-number" data-line-number="1776"></td>
        <td id="LC1776" class="blob-code blob-code-inner js-file-line">        xix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__xi_x()</td>
      </tr>
      <tr>
        <td id="L1777" class="blob-num js-line-number" data-line-number="1777"></td>
        <td id="LC1777" class="blob-code blob-code-inner js-file-line">        derxix <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_xi_x()</td>
      </tr>
      <tr>
        <td id="L1778" class="blob-num js-line-number" data-line-number="1778"></td>
        <td id="LC1778" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1779" class="blob-num js-line-number" data-line-number="1779"></td>
        <td id="LC1779" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span>xix <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">-</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>) <span class="pl-k">+</span> <span class="pl-c1">pow</span>(xix,<span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L1780" class="blob-num js-line-number" data-line-number="1780"></td>
        <td id="LC1780" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">4</span> <span class="pl-k">*</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">3</span>) <span class="pl-k">/</span> den</td>
      </tr>
      <tr>
        <td id="L1781" class="blob-num js-line-number" data-line-number="1781"></td>
        <td id="LC1781" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>xix,<span class="pl-c1">4</span>) <span class="pl-k">*</span> (<span class="pl-c1">4</span> <span class="pl-k">+</span> <span class="pl-c1">8</span><span class="pl-k">*</span>xix <span class="pl-k">-</span><span class="pl-c1">12</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">2</span>) <span class="pl-k">+</span> <span class="pl-c1">4</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(xix,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>(den,<span class="pl-c1">2</span>)</td>
      </tr>
      <tr>
        <td id="L1782" class="blob-num js-line-number" data-line-number="1782"></td>
        <td id="LC1782" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> derxix <span class="pl-k">*</span> <span class="pl-k">-</span>(t1 <span class="pl-k">+</span> t2)</td>
      </tr>
      <tr>
        <td id="L1783" class="blob-num js-line-number" data-line-number="1783"></td>
        <td id="LC1783" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1784" class="blob-num js-line-number" data-line-number="1784"></td>
        <td id="LC1784" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1785" class="blob-num js-line-number" data-line-number="1785"></td>
        <td id="LC1785" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1786" class="blob-num js-line-number" data-line-number="1786"></td>
        <td id="LC1786" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__der_a2kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">gcomp</span>, <span class="pl-smi">hsd</span>):</td>
      </tr>
      <tr>
        <td id="L1787" class="blob-num js-line-number" data-line-number="1787"></td>
        <td id="LC1787" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L1788" class="blob-num js-line-number" data-line-number="1788"></td>
        <td id="LC1788" class="blob-code blob-code-inner js-file-line">        derkhs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__der_khs()</td>
      </tr>
      <tr>
        <td id="L1789" class="blob-num js-line-number" data-line-number="1789"></td>
        <td id="LC1789" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1790" class="blob-num js-line-number" data-line-number="1790"></td>
        <td id="LC1790" class="blob-code blob-code-inner js-file-line">        ckl <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1791" class="blob-num js-line-number" data-line-number="1791"></td>
        <td id="LC1791" class="blob-code blob-code-inner js-file-line">        x0kl <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1792" class="blob-num js-line-number" data-line-number="1792"></td>
        <td id="LC1792" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1793" class="blob-num js-line-number" data-line-number="1793"></td>
        <td id="LC1793" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1794" class="blob-num js-line-number" data-line-number="1794"></td>
        <td id="LC1794" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1795" class="blob-num js-line-number" data-line-number="1795"></td>
        <td id="LC1795" class="blob-code blob-code-inner js-file-line">        t11 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L1796" class="blob-num js-line-number" data-line-number="1796"></td>
        <td id="LC1796" class="blob-code blob-code-inner js-file-line">        t12 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L1797" class="blob-num js-line-number" data-line-number="1797"></td>
        <td id="LC1797" class="blob-code blob-code-inner js-file-line">        t13 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L1798" class="blob-num js-line-number" data-line-number="1798"></td>
        <td id="LC1798" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span>  derkhs <span class="pl-k">*</span> (t11 <span class="pl-k">-</span> t12 <span class="pl-k">+</span> t13)</td>
      </tr>
      <tr>
        <td id="L1799" class="blob-num js-line-number" data-line-number="1799"></td>
        <td id="LC1799" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1800" class="blob-num js-line-number" data-line-number="1800"></td>
        <td id="LC1800" class="blob-code blob-code-inner js-file-line">        t21 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>att))</td>
      </tr>
      <tr>
        <td id="L1801" class="blob-num js-line-number" data-line-number="1801"></td>
        <td id="LC1801" class="blob-code blob-code-inner js-file-line">        t22 <span class="pl-k">=</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">pow</span>(x0kl, att<span class="pl-k">+</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(gcomp, hsd, att<span class="pl-k">+</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(gcomp, hsd, att<span class="pl-k">+</span>rep))</td>
      </tr>
      <tr>
        <td id="L1802" class="blob-num js-line-number" data-line-number="1802"></td>
        <td id="LC1802" class="blob-code blob-code-inner js-file-line">        t23 <span class="pl-k">=</span> <span class="pl-c1">pow</span>(x0kl, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__der_as1kl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__der_bkl(gcomp, hsd, <span class="pl-c1">2</span><span class="pl-k">*</span>rep))</td>
      </tr>
      <tr>
        <td id="L1803" class="blob-num js-line-number" data-line-number="1803"></td>
        <td id="LC1803" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> khs <span class="pl-k">*</span> (t21 <span class="pl-k">-</span> t22 <span class="pl-k">+</span> t23)</td>
      </tr>
      <tr>
        <td id="L1804" class="blob-num js-line-number" data-line-number="1804"></td>
        <td id="LC1804" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1805" class="blob-num js-line-number" data-line-number="1805"></td>
        <td id="LC1805" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">0.5</span> <span class="pl-k">*</span> epsi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(ckl, <span class="pl-c1">2</span>) <span class="pl-k">*</span> (t1 <span class="pl-k">+</span> t2)</td>
      </tr>
      <tr>
        <td id="L1806" class="blob-num js-line-number" data-line-number="1806"></td>
        <td id="LC1806" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1807" class="blob-num js-line-number" data-line-number="1807"></td>
        <td id="LC1807" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1808" class="blob-num js-line-number" data-line-number="1808"></td>
        <td id="LC1808" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">__g2mca</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">ci</span>):</td>
      </tr>
      <tr>
        <td id="L1809" class="blob-num js-line-number" data-line-number="1809"></td>
        <td id="LC1809" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1810" class="blob-num js-line-number" data-line-number="1810"></td>
        <td id="LC1810" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        1 / (2pi epsiii^2 hsdii^3) *</span></td>
      </tr>
      <tr>
        <td id="L1811" class="blob-num js-line-number" data-line-number="1811"></td>
        <td id="LC1811" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        [</span></td>
      </tr>
      <tr>
        <td id="L1812" class="blob-num js-line-number" data-line-number="1812"></td>
        <td id="LC1812" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        3 * d/dp (a2ii/(1+chi))</span></td>
      </tr>
      <tr>
        <td id="L1813" class="blob-num js-line-number" data-line-number="1813"></td>
        <td id="LC1813" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - epsiii * KHS * premie^2 * rep * x0ii^2rep * (as1kl(2rep) + Bkl(2rep))/segden </span></td>
      </tr>
      <tr>
        <td id="L1814" class="blob-num js-line-number" data-line-number="1814"></td>
        <td id="LC1814" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        + epsiii * KHS * premie^2 * (rep+att) * x0ii^(rep+att) * (as1kl(rep+att) + Bkl(rep+att))/segden </span></td>
      </tr>
      <tr>
        <td id="L1815" class="blob-num js-line-number" data-line-number="1815"></td>
        <td id="LC1815" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        - epsiii * KHS * premie^2 * att * x0ii^2att * (as1kl(2att) + Bkl(2att))/segden </span></td>
      </tr>
      <tr>
        <td id="L1816" class="blob-num js-line-number" data-line-number="1816"></td>
        <td id="LC1816" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        ]</span></td>
      </tr>
      <tr>
        <td id="L1817" class="blob-num js-line-number" data-line-number="1817"></td>
        <td id="LC1817" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1818" class="blob-num js-line-number" data-line-number="1818"></td>
        <td id="LC1818" class="blob-code blob-code-inner js-file-line">        gcomp <span class="pl-k">=</span> <span class="pl-c1">self</span>.ccomb[ci]</td>
      </tr>
      <tr>
        <td id="L1819" class="blob-num js-line-number" data-line-number="1819"></td>
        <td id="LC1819" class="blob-code blob-code-inner js-file-line">        hsd <span class="pl-k">=</span> <span class="pl-c1">self</span>.chsd[ci]</td>
      </tr>
      <tr>
        <td id="L1820" class="blob-num js-line-number" data-line-number="1820"></td>
        <td id="LC1820" class="blob-code blob-code-inner js-file-line">        premie <span class="pl-k">=</span> gcomp.premie()</td>
      </tr>
      <tr>
        <td id="L1821" class="blob-num js-line-number" data-line-number="1821"></td>
        <td id="LC1821" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> gcomp.epsilon <span class="pl-k">*</span> cst.k</td>
      </tr>
      <tr>
        <td id="L1822" class="blob-num js-line-number" data-line-number="1822"></td>
        <td id="LC1822" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> gcomp.rep</td>
      </tr>
      <tr>
        <td id="L1823" class="blob-num js-line-number" data-line-number="1823"></td>
        <td id="LC1823" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> gcomp.att</td>
      </tr>
      <tr>
        <td id="L1824" class="blob-num js-line-number" data-line-number="1824"></td>
        <td id="LC1824" class="blob-code blob-code-inner js-file-line">        x0ii <span class="pl-k">=</span> gcomp.x0kl(hsd)</td>
      </tr>
      <tr>
        <td id="L1825" class="blob-num js-line-number" data-line-number="1825"></td>
        <td id="LC1825" class="blob-code blob-code-inner js-file-line">        khs <span class="pl-k">=</span> <span class="pl-c1">self</span>.__khs()</td>
      </tr>
      <tr>
        <td id="L1826" class="blob-num js-line-number" data-line-number="1826"></td>
        <td id="LC1826" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1827" class="blob-num js-line-number" data-line-number="1827"></td>
        <td id="LC1827" class="blob-code blob-code-inner js-file-line">        segden <span class="pl-k">=</span> <span class="pl-c1">self</span>.segden</td>
      </tr>
      <tr>
        <td id="L1828" class="blob-num js-line-number" data-line-number="1828"></td>
        <td id="LC1828" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1829" class="blob-num js-line-number" data-line-number="1829"></td>
        <td id="LC1829" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">*</span> <span class="pl-c1">self</span>.__der_a2kl(gcomp, hsd)</td>
      </tr>
      <tr>
        <td id="L1830" class="blob-num js-line-number" data-line-number="1830"></td>
        <td id="LC1830" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> rep <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, <span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd,<span class="pl-c1">2</span><span class="pl-k">*</span>rep) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd,<span class="pl-c1">2</span><span class="pl-k">*</span>rep)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1831" class="blob-num js-line-number" data-line-number="1831"></td>
        <td id="LC1831" class="blob-code blob-code-inner js-file-line">        t3 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> (rep<span class="pl-k">+</span>att) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, rep<span class="pl-k">+</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd,rep<span class="pl-k">+</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd,rep<span class="pl-k">+</span>att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1832" class="blob-num js-line-number" data-line-number="1832"></td>
        <td id="LC1832" class="blob-code blob-code-inner js-file-line">        t4 <span class="pl-k">=</span> epsi <span class="pl-k">*</span> khs <span class="pl-k">*</span> <span class="pl-c1">pow</span>(premie,<span class="pl-c1">2</span>) <span class="pl-k">*</span> att <span class="pl-k">*</span> <span class="pl-c1">pow</span>(x0ii, <span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">*</span> (<span class="pl-c1">self</span>.__as1kl(gcomp, hsd,<span class="pl-c1">2</span><span class="pl-k">*</span>att) <span class="pl-k">+</span> <span class="pl-c1">self</span>.__bkl(gcomp, hsd,<span class="pl-c1">2</span><span class="pl-k">*</span>att)) <span class="pl-k">/</span> segden</td>
      </tr>
      <tr>
        <td id="L1833" class="blob-num js-line-number" data-line-number="1833"></td>
        <td id="LC1833" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L1834" class="blob-num js-line-number" data-line-number="1834"></td>
        <td id="LC1834" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">/</span> (<span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> <span class="pl-c1">pow</span>(epsi,<span class="pl-c1">2</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(hsd, <span class="pl-c1">3</span>)) <span class="pl-k">*</span> (t1 <span class="pl-k">-</span> t2 <span class="pl-k">+</span> t3 <span class="pl-k">-</span> t4)</td>
      </tr>
      <tr>
        <td id="L1835" class="blob-num js-line-number" data-line-number="1835"></td>
        <td id="LC1835" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1836" class="blob-num js-line-number" data-line-number="1836"></td>
        <td id="LC1836" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1837" class="blob-num js-line-number" data-line-number="1837"></td>
        <td id="LC1837" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1838" class="blob-num js-line-number" data-line-number="1838"></td>
        <td id="LC1838" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for assoc</span></td>
      </tr>
      <tr>
        <td id="L1839" class="blob-num js-line-number" data-line-number="1839"></td>
        <td id="LC1839" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1840" class="blob-num js-line-number" data-line-number="1840"></td>
        <td id="LC1840" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1841" class="blob-num js-line-number" data-line-number="1841"></td>
        <td id="LC1841" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">MieGroup</span>(<span class="pl-c1">object</span>):</td>
      </tr>
      <tr>
        <td id="L1842" class="blob-num js-line-number" data-line-number="1842"></td>
        <td id="LC1842" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">lambda_r</span>, <span class="pl-smi">lambda_a</span>, <span class="pl-smi">sigma</span>, <span class="pl-smi">epsilon</span>):</td>
      </tr>
      <tr>
        <td id="L1843" class="blob-num js-line-number" data-line-number="1843"></td>
        <td id="LC1843" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1844" class="blob-num js-line-number" data-line-number="1844"></td>
        <td id="LC1844" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Init for basic Mie group with Mie parameters</span></td>
      </tr>
      <tr>
        <td id="L1845" class="blob-num js-line-number" data-line-number="1845"></td>
        <td id="LC1845" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1846" class="blob-num js-line-number" data-line-number="1846"></td>
        <td id="LC1846" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.rep <span class="pl-k">=</span> lambda_r</td>
      </tr>
      <tr>
        <td id="L1847" class="blob-num js-line-number" data-line-number="1847"></td>
        <td id="LC1847" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.att <span class="pl-k">=</span> lambda_a</td>
      </tr>
      <tr>
        <td id="L1848" class="blob-num js-line-number" data-line-number="1848"></td>
        <td id="LC1848" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.sigma <span class="pl-k">=</span> sigma <span class="pl-c"><span class="pl-c">#</span> units A</span></td>
      </tr>
      <tr>
        <td id="L1849" class="blob-num js-line-number" data-line-number="1849"></td>
        <td id="LC1849" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.epsilon <span class="pl-k">=</span> epsilon <span class="pl-c"><span class="pl-c">#</span> units K</span></td>
      </tr>
      <tr>
        <td id="L1850" class="blob-num js-line-number" data-line-number="1850"></td>
        <td id="LC1850" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1851" class="blob-num js-line-number" data-line-number="1851"></td>
        <td id="LC1851" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__add__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>):</td>
      </tr>
      <tr>
        <td id="L1852" class="blob-num js-line-number" data-line-number="1852"></td>
        <td id="LC1852" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1853" class="blob-num js-line-number" data-line-number="1853"></td>
        <td id="LC1853" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Basic combination rules for Mie groups in general</span></td>
      </tr>
      <tr>
        <td id="L1854" class="blob-num js-line-number" data-line-number="1854"></td>
        <td id="LC1854" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1855" class="blob-num js-line-number" data-line-number="1855"></td>
        <td id="LC1855" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, MieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>MieGroup can only combine with other MieGroup<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L1856" class="blob-num js-line-number" data-line-number="1856"></td>
        <td id="LC1856" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> other <span class="pl-k">is</span> <span class="pl-c1">self</span>:</td>
      </tr>
      <tr>
        <td id="L1857" class="blob-num js-line-number" data-line-number="1857"></td>
        <td id="LC1857" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> MieGroup(<span class="pl-c1">self</span>.rep, <span class="pl-c1">self</span>.att, <span class="pl-c1">self</span>.sigma, <span class="pl-c1">self</span>.epsilon)</td>
      </tr>
      <tr>
        <td id="L1858" class="blob-num js-line-number" data-line-number="1858"></td>
        <td id="LC1858" class="blob-code blob-code-inner js-file-line">        sigma <span class="pl-k">=</span> (<span class="pl-c1">self</span>.sigma <span class="pl-k">+</span> other.sigma) <span class="pl-k">/</span> <span class="pl-c1">2</span></td>
      </tr>
      <tr>
        <td id="L1859" class="blob-num js-line-number" data-line-number="1859"></td>
        <td id="LC1859" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> sqrt(<span class="pl-c1">self</span>.sigma<span class="pl-k">**</span><span class="pl-c1">3</span> <span class="pl-k">*</span> other.sigma<span class="pl-k">**</span><span class="pl-c1">3</span>) <span class="pl-k">/</span> sigma<span class="pl-k">**</span><span class="pl-c1">3</span> <span class="pl-k">*</span> sqrt(<span class="pl-c1">self</span>.epsilon <span class="pl-k">*</span> other.epsilon)</td>
      </tr>
      <tr>
        <td id="L1860" class="blob-num js-line-number" data-line-number="1860"></td>
        <td id="LC1860" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> sqrt((<span class="pl-c1">self</span>.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (other.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L1861" class="blob-num js-line-number" data-line-number="1861"></td>
        <td id="LC1861" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> sqrt((<span class="pl-c1">self</span>.att <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (other.att <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L1862" class="blob-num js-line-number" data-line-number="1862"></td>
        <td id="LC1862" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1863" class="blob-num js-line-number" data-line-number="1863"></td>
        <td id="LC1863" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> MieGroup(rep, att, sigma, epsi)</td>
      </tr>
      <tr>
        <td id="L1864" class="blob-num js-line-number" data-line-number="1864"></td>
        <td id="LC1864" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1865" class="blob-num js-line-number" data-line-number="1865"></td>
        <td id="LC1865" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1866" class="blob-num js-line-number" data-line-number="1866"></td>
        <td id="LC1866" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1867" class="blob-num js-line-number" data-line-number="1867"></td>
        <td id="LC1867" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__repr__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1868" class="blob-num js-line-number" data-line-number="1868"></td>
        <td id="LC1868" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-s"><span class="pl-pds">&quot;</span>&lt;Mie-(<span class="pl-c1">{<span class="pl-k">:5.3f</span>}</span>,<span class="pl-c1">{<span class="pl-k">:5.3f</span>}</span>) at (<span class="pl-c1">{<span class="pl-k">:4.3f</span>}</span> a, <span class="pl-c1">{<span class="pl-k">:4.3f</span>}</span> K)&gt;<span class="pl-pds">&quot;</span></span>.format(<span class="pl-c1">self</span>.rep, <span class="pl-c1">self</span>.att, <span class="pl-c1">self</span>.sigma, <span class="pl-c1">self</span>.epsilon)</td>
      </tr>
      <tr>
        <td id="L1869" class="blob-num js-line-number" data-line-number="1869"></td>
        <td id="LC1869" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1870" class="blob-num js-line-number" data-line-number="1870"></td>
        <td id="LC1870" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">premie</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1871" class="blob-num js-line-number" data-line-number="1871"></td>
        <td id="LC1871" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.rep<span class="pl-k">/</span>(<span class="pl-c1">self</span>.rep<span class="pl-k">-</span><span class="pl-c1">self</span>.att) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(<span class="pl-c1">self</span>.rep<span class="pl-k">/</span><span class="pl-c1">self</span>.att, <span class="pl-c1">self</span>.att<span class="pl-k">/</span>(<span class="pl-c1">self</span>.rep<span class="pl-k">-</span><span class="pl-c1">self</span>.att))</td>
      </tr>
      <tr>
        <td id="L1872" class="blob-num js-line-number" data-line-number="1872"></td>
        <td id="LC1872" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1873" class="blob-num js-line-number" data-line-number="1873"></td>
        <td id="LC1873" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">hsd</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temperature</span>, <span class="pl-smi">x_inf</span><span class="pl-k">=</span><span class="pl-c1">0</span>.):</td>
      </tr>
      <tr>
        <td id="L1874" class="blob-num js-line-number" data-line-number="1874"></td>
        <td id="LC1874" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1875" class="blob-num js-line-number" data-line-number="1875"></td>
        <td id="LC1875" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Calculates hard sphere diameter using the method defined in methods.py</span></td>
      </tr>
      <tr>
        <td id="L1876" class="blob-num js-line-number" data-line-number="1876"></td>
        <td id="LC1876" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Temperature takes in as units of K</span></td>
      </tr>
      <tr>
        <td id="L1877" class="blob-num js-line-number" data-line-number="1877"></td>
        <td id="LC1877" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1878" class="blob-num js-line-number" data-line-number="1878"></td>
        <td id="LC1878" class="blob-code blob-code-inner js-file-line">        result <span class="pl-k">=</span> mt.hsdiam(temperature <span class="pl-k">/</span> <span class="pl-c1">self</span>.epsilon, <span class="pl-c1">self</span>.rep, <span class="pl-c1">self</span>.att, x_inf) <span class="pl-k">*</span> <span class="pl-c1">self</span>.sigma</td>
      </tr>
      <tr>
        <td id="L1879" class="blob-num js-line-number" data-line-number="1879"></td>
        <td id="LC1879" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> result</td>
      </tr>
      <tr>
        <td id="L1880" class="blob-num js-line-number" data-line-number="1880"></td>
        <td id="LC1880" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1881" class="blob-num js-line-number" data-line-number="1881"></td>
        <td id="LC1881" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">x0kl</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">hsdval</span>):</td>
      </tr>
      <tr>
        <td id="L1882" class="blob-num js-line-number" data-line-number="1882"></td>
        <td id="LC1882" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span>.sigma <span class="pl-k">*</span> cst.atom <span class="pl-k">/</span> hsdval <span class="pl-c"><span class="pl-c">#</span> sigma in angstrom, hsd given in SI</span></td>
      </tr>
      <tr>
        <td id="L1883" class="blob-num js-line-number" data-line-number="1883"></td>
        <td id="LC1883" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1884" class="blob-num js-line-number" data-line-number="1884"></td>
        <td id="LC1884" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">Component</span>(<span class="pl-c1">object</span>):</td>
      </tr>
      <tr>
        <td id="L1885" class="blob-num js-line-number" data-line-number="1885"></td>
        <td id="LC1885" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">molar_weight</span>):</td>
      </tr>
      <tr>
        <td id="L1886" class="blob-num js-line-number" data-line-number="1886"></td>
        <td id="LC1886" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.mw <span class="pl-k">=</span> molar_weight</td>
      </tr>
      <tr>
        <td id="L1887" class="blob-num js-line-number" data-line-number="1887"></td>
        <td id="LC1887" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L1888" class="blob-num js-line-number" data-line-number="1888"></td>
        <td id="LC1888" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L1889" class="blob-num js-line-number" data-line-number="1889"></td>
        <td id="LC1889" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1890" class="blob-num js-line-number" data-line-number="1890"></td>
        <td id="LC1890" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">thdebroglie</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">temp</span>):</td>
      </tr>
      <tr>
        <td id="L1891" class="blob-num js-line-number" data-line-number="1891"></td>
        <td id="LC1891" class="blob-code blob-code-inner js-file-line">        <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1892" class="blob-num js-line-number" data-line-number="1892"></td>
        <td id="LC1892" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        Takes in component mass (au) and temperature (K) and return thermal de broglie wavelength</span></td>
      </tr>
      <tr>
        <td id="L1893" class="blob-num js-line-number" data-line-number="1893"></td>
        <td id="LC1893" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1894" class="blob-num js-line-number" data-line-number="1894"></td>
        <td id="LC1894" class="blob-code blob-code-inner js-file-line">        Lambda_sq <span class="pl-k">=</span> <span class="pl-c1">pow</span>(cst.h,<span class="pl-c1">2</span>) <span class="pl-k">*</span> <span class="pl-c1">1e3</span> <span class="pl-k">*</span> cst.Na <span class="pl-k">/</span> (<span class="pl-c1">2</span> <span class="pl-k">*</span> pi <span class="pl-k">*</span> <span class="pl-c1">self</span>.mw <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> temp)</td>
      </tr>
      <tr>
        <td id="L1895" class="blob-num js-line-number" data-line-number="1895"></td>
        <td id="LC1895" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> sqrt(Lambda_sq)</td>
      </tr>
      <tr>
        <td id="L1896" class="blob-num js-line-number" data-line-number="1896"></td>
        <td id="LC1896" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1897" class="blob-num js-line-number" data-line-number="1897"></td>
        <td id="LC1897" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cp_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">cpparam</span>, <span class="pl-smi">ref</span><span class="pl-k">=</span>(<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.,<span class="pl-c1">273.15</span>,<span class="pl-c1">101325</span>)):</td>
      </tr>
      <tr>
        <td id="L1898" class="blob-num js-line-number" data-line-number="1898"></td>
        <td id="LC1898" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(cpparam, (np.ndarray, <span class="pl-c1">list</span>, <span class="pl-c1">tuple</span>)) <span class="pl-k">and</span> <span class="pl-c1">len</span>(cpparam)<span class="pl-k">==</span><span class="pl-c1">4</span>,</td>
      </tr>
      <tr>
        <td id="L1899" class="blob-num js-line-number" data-line-number="1899"></td>
        <td id="LC1899" class="blob-code blob-code-inner js-file-line">            <span class="pl-s"><span class="pl-pds">&quot;</span>Cp parameters must be iterable with length of 4<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L1900" class="blob-num js-line-number" data-line-number="1900"></td>
        <td id="LC1900" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Cp parameters in form [A, B, C, D]</span></td>
      </tr>
      <tr>
        <td id="L1901" class="blob-num js-line-number" data-line-number="1901"></td>
        <td id="LC1901" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> cpparam</td>
      </tr>
      <tr>
        <td id="L1902" class="blob-num js-line-number" data-line-number="1902"></td>
        <td id="LC1902" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> ref <span class="pl-c"><span class="pl-c">#</span> ref: h0, s0, T0, P0</span></td>
      </tr>
      <tr>
        <td id="L1903" class="blob-num js-line-number" data-line-number="1903"></td>
        <td id="LC1903" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1904" class="blob-num js-line-number" data-line-number="1904"></td>
        <td id="LC1904" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">cp_int</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">T</span>):</td>
      </tr>
      <tr>
        <td id="L1905" class="blob-num js-line-number" data-line-number="1905"></td>
        <td id="LC1905" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> cp integral evaluated at T</span></td>
      </tr>
      <tr>
        <td id="L1906" class="blob-num js-line-number" data-line-number="1906"></td>
        <td id="LC1906" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> A(T-Tref) + B/2(T^2-Tref^2) + C/3(T^3-Tref^3) + D/4(T^4-Tref^4)</span></td>
      </tr>
      <tr>
        <td id="L1907" class="blob-num js-line-number" data-line-number="1907"></td>
        <td id="LC1907" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>: <span class="pl-k">return</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L1908" class="blob-num js-line-number" data-line-number="1908"></td>
        <td id="LC1908" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1909" class="blob-num js-line-number" data-line-number="1909"></td>
        <td id="LC1909" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L1910" class="blob-num js-line-number" data-line-number="1910"></td>
        <td id="LC1910" class="blob-code blob-code-inner js-file-line">        B <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L1911" class="blob-num js-line-number" data-line-number="1911"></td>
        <td id="LC1911" class="blob-code blob-code-inner js-file-line">        C <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L1912" class="blob-num js-line-number" data-line-number="1912"></td>
        <td id="LC1912" class="blob-code blob-code-inner js-file-line">        D <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">3</span>]</td>
      </tr>
      <tr>
        <td id="L1913" class="blob-num js-line-number" data-line-number="1913"></td>
        <td id="LC1913" class="blob-code blob-code-inner js-file-line">        tref <span class="pl-k">=</span> <span class="pl-c1">self</span>.ref[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L1914" class="blob-num js-line-number" data-line-number="1914"></td>
        <td id="LC1914" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1915" class="blob-num js-line-number" data-line-number="1915"></td>
        <td id="LC1915" class="blob-code blob-code-inner js-file-line">        cpint <span class="pl-k">=</span> A<span class="pl-k">*</span>T <span class="pl-k">+</span> (B<span class="pl-k">/</span><span class="pl-c1">2</span>)<span class="pl-k">*</span>T<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">+</span> (C<span class="pl-k">/</span><span class="pl-c1">3</span>)<span class="pl-k">*</span>T<span class="pl-k">**</span><span class="pl-c1">3</span> <span class="pl-k">+</span> (D<span class="pl-k">/</span><span class="pl-c1">4</span>)<span class="pl-k">*</span>T<span class="pl-k">**</span><span class="pl-c1">4</span> <span class="pl-k">-</span> (A<span class="pl-k">*</span>tref <span class="pl-k">+</span> (B<span class="pl-k">/</span><span class="pl-c1">2</span>)<span class="pl-k">*</span>tref<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">+</span> (C<span class="pl-k">/</span><span class="pl-c1">3</span>)<span class="pl-k">*</span>tref<span class="pl-k">**</span><span class="pl-c1">3</span> <span class="pl-k">+</span> (D<span class="pl-k">/</span><span class="pl-c1">4</span>)<span class="pl-k">*</span>tref<span class="pl-k">**</span><span class="pl-c1">4</span>)</td>
      </tr>
      <tr>
        <td id="L1916" class="blob-num js-line-number" data-line-number="1916"></td>
        <td id="LC1916" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> cpint</td>
      </tr>
      <tr>
        <td id="L1917" class="blob-num js-line-number" data-line-number="1917"></td>
        <td id="LC1917" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1918" class="blob-num js-line-number" data-line-number="1918"></td>
        <td id="LC1918" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">cp_t_int</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">T</span>):</td>
      </tr>
      <tr>
        <td id="L1919" class="blob-num js-line-number" data-line-number="1919"></td>
        <td id="LC1919" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> cp/T integral evaluated at T</span></td>
      </tr>
      <tr>
        <td id="L1920" class="blob-num js-line-number" data-line-number="1920"></td>
        <td id="LC1920" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> A (ln(T)-ln(Tref)) + B(T-Tref) + C/2(T^2-Tref^2) + D/3(T^3-Tref^3)</span></td>
      </tr>
      <tr>
        <td id="L1921" class="blob-num js-line-number" data-line-number="1921"></td>
        <td id="LC1921" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>: <span class="pl-k">return</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L1922" class="blob-num js-line-number" data-line-number="1922"></td>
        <td id="LC1922" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1923" class="blob-num js-line-number" data-line-number="1923"></td>
        <td id="LC1923" class="blob-code blob-code-inner js-file-line">        A <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">0</span>]</td>
      </tr>
      <tr>
        <td id="L1924" class="blob-num js-line-number" data-line-number="1924"></td>
        <td id="LC1924" class="blob-code blob-code-inner js-file-line">        B <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L1925" class="blob-num js-line-number" data-line-number="1925"></td>
        <td id="LC1925" class="blob-code blob-code-inner js-file-line">        C <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L1926" class="blob-num js-line-number" data-line-number="1926"></td>
        <td id="LC1926" class="blob-code blob-code-inner js-file-line">        D <span class="pl-k">=</span> <span class="pl-c1">self</span>.cp[<span class="pl-c1">3</span>]</td>
      </tr>
      <tr>
        <td id="L1927" class="blob-num js-line-number" data-line-number="1927"></td>
        <td id="LC1927" class="blob-code blob-code-inner js-file-line">        tref <span class="pl-k">=</span> <span class="pl-c1">self</span>.ref[<span class="pl-c1">2</span>]</td>
      </tr>
      <tr>
        <td id="L1928" class="blob-num js-line-number" data-line-number="1928"></td>
        <td id="LC1928" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1929" class="blob-num js-line-number" data-line-number="1929"></td>
        <td id="LC1929" class="blob-code blob-code-inner js-file-line">        cptint <span class="pl-k">=</span> A<span class="pl-k">*</span>log(T) <span class="pl-k">+</span> B<span class="pl-k">*</span>T <span class="pl-k">+</span> (C<span class="pl-k">/</span><span class="pl-c1">2</span>)<span class="pl-k">*</span>T<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">+</span> (D<span class="pl-k">/</span><span class="pl-c1">3</span>)<span class="pl-k">*</span>T<span class="pl-k">**</span><span class="pl-c1">3</span> <span class="pl-k">-</span> (A<span class="pl-k">*</span>log(tref) <span class="pl-k">+</span> B<span class="pl-k">*</span>tref <span class="pl-k">+</span> (C<span class="pl-k">/</span><span class="pl-c1">2</span>)<span class="pl-k">*</span>tref<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">+</span> (D<span class="pl-k">/</span><span class="pl-c1">3</span>)<span class="pl-k">*</span>tref<span class="pl-k">**</span><span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1930" class="blob-num js-line-number" data-line-number="1930"></td>
        <td id="LC1930" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> cptint</td>
      </tr>
      <tr>
        <td id="L1931" class="blob-num js-line-number" data-line-number="1931"></td>
        <td id="LC1931" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1932" class="blob-num js-line-number" data-line-number="1932"></td>
        <td id="LC1932" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">VRMieComponent</span>(<span class="pl-e">MieGroup</span>, <span class="pl-e">Component</span>):</td>
      </tr>
      <tr>
        <td id="L1933" class="blob-num js-line-number" data-line-number="1933"></td>
        <td id="LC1933" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1934" class="blob-num js-line-number" data-line-number="1934"></td>
        <td id="LC1934" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    SAFT-VR Mie Component as defined by SAFT-VR Mie Equation-of-state</span></td>
      </tr>
      <tr>
        <td id="L1935" class="blob-num js-line-number" data-line-number="1935"></td>
        <td id="LC1935" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    given in ref: J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786</span></td>
      </tr>
      <tr>
        <td id="L1936" class="blob-num js-line-number" data-line-number="1936"></td>
        <td id="LC1936" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L1937" class="blob-num js-line-number" data-line-number="1937"></td>
        <td id="LC1937" class="blob-code blob-code-inner js-file-line">    _total <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L1938" class="blob-num js-line-number" data-line-number="1938"></td>
        <td id="LC1938" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">mspheres</span>, <span class="pl-smi">mw</span>, <span class="pl-smi">lambda_r</span>, <span class="pl-smi">lambda_a</span>, <span class="pl-smi">sigma</span>, <span class="pl-smi">epsilon</span>):</td>
      </tr>
      <tr>
        <td id="L1939" class="blob-num js-line-number" data-line-number="1939"></td>
        <td id="LC1939" class="blob-code blob-code-inner js-file-line">        MieGroup.<span class="pl-c1">__init__</span>(<span class="pl-c1">self</span>, lambda_r, lambda_a, sigma, epsilon)</td>
      </tr>
      <tr>
        <td id="L1940" class="blob-num js-line-number" data-line-number="1940"></td>
        <td id="LC1940" class="blob-code blob-code-inner js-file-line">        Component.<span class="pl-c1">__init__</span>(<span class="pl-c1">self</span>, mw)</td>
      </tr>
      <tr>
        <td id="L1941" class="blob-num js-line-number" data-line-number="1941"></td>
        <td id="LC1941" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ms <span class="pl-k">=</span> mspheres</td>
      </tr>
      <tr>
        <td id="L1942" class="blob-num js-line-number" data-line-number="1942"></td>
        <td id="LC1942" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.index <span class="pl-k">=</span> VRMieComponent._total</td>
      </tr>
      <tr>
        <td id="L1943" class="blob-num js-line-number" data-line-number="1943"></td>
        <td id="LC1943" class="blob-code blob-code-inner js-file-line">        VRMieComponent._total <span class="pl-k">+=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L1944" class="blob-num js-line-number" data-line-number="1944"></td>
        <td id="LC1944" class="blob-code blob-code-inner js-file-line">        VRMieComponent.add_elem()</td>
      </tr>
      <tr>
        <td id="L1945" class="blob-num js-line-number" data-line-number="1945"></td>
        <td id="LC1945" class="blob-code blob-code-inner js-file-line">        </td>
      </tr>
      <tr>
        <td id="L1946" class="blob-num js-line-number" data-line-number="1946"></td>
        <td id="LC1946" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__repr__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L1947" class="blob-num js-line-number" data-line-number="1947"></td>
        <td id="LC1947" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-s"><span class="pl-pds">&quot;</span>VR-Mie: <span class="pl-c1">{<span class="pl-k">:5.3f</span>}</span> spheres of <span class="pl-pds">&quot;</span></span>.format(<span class="pl-c1">self</span>.ms) <span class="pl-k">+</span> <span class="pl-c1">super</span>().<span class="pl-c1">__repr__</span>()</td>
      </tr>
      <tr>
        <td id="L1948" class="blob-num js-line-number" data-line-number="1948"></td>
        <td id="LC1948" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1949" class="blob-num js-line-number" data-line-number="1949"></td>
        <td id="LC1949" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Combining rules with exceptions build-in</span></td>
      </tr>
      <tr>
        <td id="L1950" class="blob-num js-line-number" data-line-number="1950"></td>
        <td id="LC1950" class="blob-code blob-code-inner js-file-line">    _ktable <span class="pl-k">=</span> np.array([[<span class="pl-c1">0</span>.]])</td>
      </tr>
      <tr>
        <td id="L1951" class="blob-num js-line-number" data-line-number="1951"></td>
        <td id="LC1951" class="blob-code blob-code-inner js-file-line">    _gtable <span class="pl-k">=</span> np.array([[<span class="pl-c1">0</span>.]])</td>
      </tr>
      <tr>
        <td id="L1952" class="blob-num js-line-number" data-line-number="1952"></td>
        <td id="LC1952" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1953" class="blob-num js-line-number" data-line-number="1953"></td>
        <td id="LC1953" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L1954" class="blob-num js-line-number" data-line-number="1954"></td>
        <td id="LC1954" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">n_total</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L1955" class="blob-num js-line-number" data-line-number="1955"></td>
        <td id="LC1955" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">cls</span>._total</td>
      </tr>
      <tr>
        <td id="L1956" class="blob-num js-line-number" data-line-number="1956"></td>
        <td id="LC1956" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1957" class="blob-num js-line-number" data-line-number="1957"></td>
        <td id="LC1957" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L1958" class="blob-num js-line-number" data-line-number="1958"></td>
        <td id="LC1958" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">add_elem</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L1959" class="blob-num js-line-number" data-line-number="1959"></td>
        <td id="LC1959" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">cls</span>._ktable</td>
      </tr>
      <tr>
        <td id="L1960" class="blob-num js-line-number" data-line-number="1960"></td>
        <td id="LC1960" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">cls</span>._gtable</td>
      </tr>
      <tr>
        <td id="L1961" class="blob-num js-line-number" data-line-number="1961"></td>
        <td id="LC1961" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">cls</span>._total <span class="pl-k">&gt;</span> <span class="pl-c1">1</span>:</td>
      </tr>
      <tr>
        <td id="L1962" class="blob-num js-line-number" data-line-number="1962"></td>
        <td id="LC1962" class="blob-code blob-code-inner js-file-line">            t12 <span class="pl-k">=</span> np.append(t1, np.zeros((<span class="pl-c1">1</span>, t1.shape[<span class="pl-c1">1</span>])), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L1963" class="blob-num js-line-number" data-line-number="1963"></td>
        <td id="LC1963" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">cls</span>._ktable <span class="pl-k">=</span> np.append(t12, np.zeros((t12.shape[<span class="pl-c1">0</span>], <span class="pl-c1">1</span>)), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L1964" class="blob-num js-line-number" data-line-number="1964"></td>
        <td id="LC1964" class="blob-code blob-code-inner js-file-line">            t22 <span class="pl-k">=</span> np.append(t2, np.zeros((<span class="pl-c1">1</span>, t2.shape[<span class="pl-c1">1</span>])), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L1965" class="blob-num js-line-number" data-line-number="1965"></td>
        <td id="LC1965" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">cls</span>._gtable <span class="pl-k">=</span> np.append(t22, np.zeros((t22.shape[<span class="pl-c1">0</span>], <span class="pl-c1">1</span>)), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L1966" class="blob-num js-line-number" data-line-number="1966"></td>
        <td id="LC1966" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1967" class="blob-num js-line-number" data-line-number="1967"></td>
        <td id="LC1967" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__add__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>):</td>
      </tr>
      <tr>
        <td id="L1968" class="blob-num js-line-number" data-line-number="1968"></td>
        <td id="LC1968" class="blob-code blob-code-inner js-file-line">        mg <span class="pl-k">=</span> <span class="pl-c1">super</span>().<span class="pl-c1">__add__</span>(other)</td>
      </tr>
      <tr>
        <td id="L1969" class="blob-num js-line-number" data-line-number="1969"></td>
        <td id="LC1969" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> VRMieComponent._ktable[<span class="pl-c1">self</span>.index, other.index]</td>
      </tr>
      <tr>
        <td id="L1970" class="blob-num js-line-number" data-line-number="1970"></td>
        <td id="LC1970" class="blob-code blob-code-inner js-file-line">        mg.epsilon <span class="pl-k">=</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> kij) <span class="pl-k">*</span> mg.epsilon</td>
      </tr>
      <tr>
        <td id="L1971" class="blob-num js-line-number" data-line-number="1971"></td>
        <td id="LC1971" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> VRMieComponent._gtable[<span class="pl-c1">self</span>.index, other.index]</td>
      </tr>
      <tr>
        <td id="L1972" class="blob-num js-line-number" data-line-number="1972"></td>
        <td id="LC1972" class="blob-code blob-code-inner js-file-line">        mg.rep <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> gij) <span class="pl-k">*</span> (mg.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L1973" class="blob-num js-line-number" data-line-number="1973"></td>
        <td id="LC1973" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> mg</td>
      </tr>
      <tr>
        <td id="L1974" class="blob-num js-line-number" data-line-number="1974"></td>
        <td id="LC1974" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1975" class="blob-num js-line-number" data-line-number="1975"></td>
        <td id="LC1975" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L1976" class="blob-num js-line-number" data-line-number="1976"></td>
        <td id="LC1976" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">print_table</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L1977" class="blob-num js-line-number" data-line-number="1977"></td>
        <td id="LC1977" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:30s</span>}{<span class="pl-k">:d</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Total VR Components: <span class="pl-pds">&quot;</span></span>, <span class="pl-c1">cls</span>._total))</td>
      </tr>
      <tr>
        <td id="L1978" class="blob-num js-line-number" data-line-number="1978"></td>
        <td id="LC1978" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:40s</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Combination k values for Epsilon:<span class="pl-pds">&quot;</span></span>))</td>
      </tr>
      <tr>
        <td id="L1979" class="blob-num js-line-number" data-line-number="1979"></td>
        <td id="LC1979" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-c1">cls</span>._ktable)</td>
      </tr>
      <tr>
        <td id="L1980" class="blob-num js-line-number" data-line-number="1980"></td>
        <td id="LC1980" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:40s</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Combination gamma values for Lambda_r:<span class="pl-pds">&quot;</span></span>))</td>
      </tr>
      <tr>
        <td id="L1981" class="blob-num js-line-number" data-line-number="1981"></td>
        <td id="LC1981" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-c1">cls</span>._gtable)</td>
      </tr>
      <tr>
        <td id="L1982" class="blob-num js-line-number" data-line-number="1982"></td>
        <td id="LC1982" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1983" class="blob-num js-line-number" data-line-number="1983"></td>
        <td id="LC1983" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L1984" class="blob-num js-line-number" data-line-number="1984"></td>
        <td id="LC1984" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_e_kij</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L1985" class="blob-num js-line-number" data-line-number="1985"></td>
        <td id="LC1985" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, VRMieComponent) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L1986" class="blob-num js-line-number" data-line-number="1986"></td>
        <td id="LC1986" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L1987" class="blob-num js-line-number" data-line-number="1987"></td>
        <td id="LC1987" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1988" class="blob-num js-line-number" data-line-number="1988"></td>
        <td id="LC1988" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L1989" class="blob-num js-line-number" data-line-number="1989"></td>
        <td id="LC1989" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L1990" class="blob-num js-line-number" data-line-number="1990"></td>
        <td id="LC1990" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L1991" class="blob-num js-line-number" data-line-number="1991"></td>
        <td id="LC1991" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L1992" class="blob-num js-line-number" data-line-number="1992"></td>
        <td id="LC1992" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_e_val</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L1993" class="blob-num js-line-number" data-line-number="1993"></td>
        <td id="LC1993" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, VRMieComponent) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L1994" class="blob-num js-line-number" data-line-number="1994"></td>
        <td id="LC1994" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L1995" class="blob-num js-line-number" data-line-number="1995"></td>
        <td id="LC1995" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L1996" class="blob-num js-line-number" data-line-number="1996"></td>
        <td id="LC1996" class="blob-code blob-code-inner js-file-line">        s1 <span class="pl-k">=</span> g1.sigma</td>
      </tr>
      <tr>
        <td id="L1997" class="blob-num js-line-number" data-line-number="1997"></td>
        <td id="LC1997" class="blob-code blob-code-inner js-file-line">        s2 <span class="pl-k">=</span> g2.sigma</td>
      </tr>
      <tr>
        <td id="L1998" class="blob-num js-line-number" data-line-number="1998"></td>
        <td id="LC1998" class="blob-code blob-code-inner js-file-line">        e1 <span class="pl-k">=</span> g1.epsilon</td>
      </tr>
      <tr>
        <td id="L1999" class="blob-num js-line-number" data-line-number="1999"></td>
        <td id="LC1999" class="blob-code blob-code-inner js-file-line">        e2 <span class="pl-k">=</span> g2.epsilon</td>
      </tr>
      <tr>
        <td id="L2000" class="blob-num js-line-number" data-line-number="2000"></td>
        <td id="LC2000" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2001" class="blob-num js-line-number" data-line-number="2001"></td>
        <td id="LC2001" class="blob-code blob-code-inner js-file-line">        actl <span class="pl-k">=</span> sqrt(<span class="pl-c1">pow</span>(s1,<span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(s2,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>((s1<span class="pl-k">+</span>s2)<span class="pl-k">/</span><span class="pl-c1">2</span>, <span class="pl-c1">3</span>) <span class="pl-k">*</span> sqrt(e1 <span class="pl-k">*</span> e2)</td>
      </tr>
      <tr>
        <td id="L2002" class="blob-num js-line-number" data-line-number="2002"></td>
        <td id="LC2002" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> val <span class="pl-k">/</span> actl</td>
      </tr>
      <tr>
        <td id="L2003" class="blob-num js-line-number" data-line-number="2003"></td>
        <td id="LC2003" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2004" class="blob-num js-line-number" data-line-number="2004"></td>
        <td id="LC2004" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2005" class="blob-num js-line-number" data-line-number="2005"></td>
        <td id="LC2005" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2006" class="blob-num js-line-number" data-line-number="2006"></td>
        <td id="LC2006" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2007" class="blob-num js-line-number" data-line-number="2007"></td>
        <td id="LC2007" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2008" class="blob-num js-line-number" data-line-number="2008"></td>
        <td id="LC2008" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_lr_gij</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2009" class="blob-num js-line-number" data-line-number="2009"></td>
        <td id="LC2009" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, VRMieComponent) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2010" class="blob-num js-line-number" data-line-number="2010"></td>
        <td id="LC2010" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2011" class="blob-num js-line-number" data-line-number="2011"></td>
        <td id="LC2011" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2012" class="blob-num js-line-number" data-line-number="2012"></td>
        <td id="LC2012" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2013" class="blob-num js-line-number" data-line-number="2013"></td>
        <td id="LC2013" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2014" class="blob-num js-line-number" data-line-number="2014"></td>
        <td id="LC2014" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2015" class="blob-num js-line-number" data-line-number="2015"></td>
        <td id="LC2015" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2016" class="blob-num js-line-number" data-line-number="2016"></td>
        <td id="LC2016" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_lr_val</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2017" class="blob-num js-line-number" data-line-number="2017"></td>
        <td id="LC2017" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, VRMieComponent) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2018" class="blob-num js-line-number" data-line-number="2018"></td>
        <td id="LC2018" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2019" class="blob-num js-line-number" data-line-number="2019"></td>
        <td id="LC2019" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2020" class="blob-num js-line-number" data-line-number="2020"></td>
        <td id="LC2020" class="blob-code blob-code-inner js-file-line">        lr1 <span class="pl-k">=</span> g1.rep</td>
      </tr>
      <tr>
        <td id="L2021" class="blob-num js-line-number" data-line-number="2021"></td>
        <td id="LC2021" class="blob-code blob-code-inner js-file-line">        lr2 <span class="pl-k">=</span> g2.rep</td>
      </tr>
      <tr>
        <td id="L2022" class="blob-num js-line-number" data-line-number="2022"></td>
        <td id="LC2022" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2023" class="blob-num js-line-number" data-line-number="2023"></td>
        <td id="LC2023" class="blob-code blob-code-inner js-file-line">        intm <span class="pl-k">=</span> val <span class="pl-k">-</span> <span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L2024" class="blob-num js-line-number" data-line-number="2024"></td>
        <td id="LC2024" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> intm <span class="pl-k">/</span> sqrt((lr1 <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (lr2 <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2025" class="blob-num js-line-number" data-line-number="2025"></td>
        <td id="LC2025" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2026" class="blob-num js-line-number" data-line-number="2026"></td>
        <td id="LC2026" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i1,i2] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2027" class="blob-num js-line-number" data-line-number="2027"></td>
        <td id="LC2027" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i2,i1] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2028" class="blob-num js-line-number" data-line-number="2028"></td>
        <td id="LC2028" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2029" class="blob-num js-line-number" data-line-number="2029"></td>
        <td id="LC2029" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_epsilon</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2030" class="blob-num js-line-number" data-line-number="2030"></td>
        <td id="LC2030" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2031" class="blob-num js-line-number" data-line-number="2031"></td>
        <td id="LC2031" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2032" class="blob-num js-line-number" data-line-number="2032"></td>
        <td id="LC2032" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2033" class="blob-num js-line-number" data-line-number="2033"></td>
        <td id="LC2033" class="blob-code blob-code-inner js-file-line">        s1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.sigma</td>
      </tr>
      <tr>
        <td id="L2034" class="blob-num js-line-number" data-line-number="2034"></td>
        <td id="LC2034" class="blob-code blob-code-inner js-file-line">        s2 <span class="pl-k">=</span> other.sigma</td>
      </tr>
      <tr>
        <td id="L2035" class="blob-num js-line-number" data-line-number="2035"></td>
        <td id="LC2035" class="blob-code blob-code-inner js-file-line">        e1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.epsilon</td>
      </tr>
      <tr>
        <td id="L2036" class="blob-num js-line-number" data-line-number="2036"></td>
        <td id="LC2036" class="blob-code blob-code-inner js-file-line">        e2 <span class="pl-k">=</span> other.epsilon</td>
      </tr>
      <tr>
        <td id="L2037" class="blob-num js-line-number" data-line-number="2037"></td>
        <td id="LC2037" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2038" class="blob-num js-line-number" data-line-number="2038"></td>
        <td id="LC2038" class="blob-code blob-code-inner js-file-line">        actl <span class="pl-k">=</span> sqrt(<span class="pl-c1">pow</span>(s1,<span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(s2,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>((s1<span class="pl-k">+</span>s2)<span class="pl-k">/</span><span class="pl-c1">2</span>, <span class="pl-c1">3</span>) <span class="pl-k">*</span> sqrt(e1 <span class="pl-k">*</span> e2)</td>
      </tr>
      <tr>
        <td id="L2039" class="blob-num js-line-number" data-line-number="2039"></td>
        <td id="LC2039" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> val <span class="pl-k">/</span> actl</td>
      </tr>
      <tr>
        <td id="L2040" class="blob-num js-line-number" data-line-number="2040"></td>
        <td id="LC2040" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2041" class="blob-num js-line-number" data-line-number="2041"></td>
        <td id="LC2041" class="blob-code blob-code-inner js-file-line">        VRMieComponent._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2042" class="blob-num js-line-number" data-line-number="2042"></td>
        <td id="LC2042" class="blob-code blob-code-inner js-file-line">        VRMieComponent._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2043" class="blob-num js-line-number" data-line-number="2043"></td>
        <td id="LC2043" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2044" class="blob-num js-line-number" data-line-number="2044"></td>
        <td id="LC2044" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> kij</td>
      </tr>
      <tr>
        <td id="L2045" class="blob-num js-line-number" data-line-number="2045"></td>
        <td id="LC2045" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2046" class="blob-num js-line-number" data-line-number="2046"></td>
        <td id="LC2046" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_e_kij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2047" class="blob-num js-line-number" data-line-number="2047"></td>
        <td id="LC2047" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2048" class="blob-num js-line-number" data-line-number="2048"></td>
        <td id="LC2048" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2049" class="blob-num js-line-number" data-line-number="2049"></td>
        <td id="LC2049" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2050" class="blob-num js-line-number" data-line-number="2050"></td>
        <td id="LC2050" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2051" class="blob-num js-line-number" data-line-number="2051"></td>
        <td id="LC2051" class="blob-code blob-code-inner js-file-line">        VRMieComponent._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2052" class="blob-num js-line-number" data-line-number="2052"></td>
        <td id="LC2052" class="blob-code blob-code-inner js-file-line">        VRMieComponent._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2053" class="blob-num js-line-number" data-line-number="2053"></td>
        <td id="LC2053" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2054" class="blob-num js-line-number" data-line-number="2054"></td>
        <td id="LC2054" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> kij</td>
      </tr>
      <tr>
        <td id="L2055" class="blob-num js-line-number" data-line-number="2055"></td>
        <td id="LC2055" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2056" class="blob-num js-line-number" data-line-number="2056"></td>
        <td id="LC2056" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_rep</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2057" class="blob-num js-line-number" data-line-number="2057"></td>
        <td id="LC2057" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2058" class="blob-num js-line-number" data-line-number="2058"></td>
        <td id="LC2058" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2059" class="blob-num js-line-number" data-line-number="2059"></td>
        <td id="LC2059" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2060" class="blob-num js-line-number" data-line-number="2060"></td>
        <td id="LC2060" class="blob-code blob-code-inner js-file-line">        lr1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.rep</td>
      </tr>
      <tr>
        <td id="L2061" class="blob-num js-line-number" data-line-number="2061"></td>
        <td id="LC2061" class="blob-code blob-code-inner js-file-line">        lr2 <span class="pl-k">=</span> other.rep</td>
      </tr>
      <tr>
        <td id="L2062" class="blob-num js-line-number" data-line-number="2062"></td>
        <td id="LC2062" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2063" class="blob-num js-line-number" data-line-number="2063"></td>
        <td id="LC2063" class="blob-code blob-code-inner js-file-line">        intm <span class="pl-k">=</span> val <span class="pl-k">-</span> <span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L2064" class="blob-num js-line-number" data-line-number="2064"></td>
        <td id="LC2064" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> intm <span class="pl-k">/</span> sqrt((lr1 <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (lr2 <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2065" class="blob-num js-line-number" data-line-number="2065"></td>
        <td id="LC2065" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2066" class="blob-num js-line-number" data-line-number="2066"></td>
        <td id="LC2066" class="blob-code blob-code-inner js-file-line">        VRMieComponent._gtable[i1,i2] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2067" class="blob-num js-line-number" data-line-number="2067"></td>
        <td id="LC2067" class="blob-code blob-code-inner js-file-line">        VRMieComponent._gtable[i2,i1] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2068" class="blob-num js-line-number" data-line-number="2068"></td>
        <td id="LC2068" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2069" class="blob-num js-line-number" data-line-number="2069"></td>
        <td id="LC2069" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> gij</td>
      </tr>
      <tr>
        <td id="L2070" class="blob-num js-line-number" data-line-number="2070"></td>
        <td id="LC2070" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2071" class="blob-num js-line-number" data-line-number="2071"></td>
        <td id="LC2071" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_r_gij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2072" class="blob-num js-line-number" data-line-number="2072"></td>
        <td id="LC2072" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, VRMieComponent), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2073" class="blob-num js-line-number" data-line-number="2073"></td>
        <td id="LC2073" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2074" class="blob-num js-line-number" data-line-number="2074"></td>
        <td id="LC2074" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2075" class="blob-num js-line-number" data-line-number="2075"></td>
        <td id="LC2075" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2076" class="blob-num js-line-number" data-line-number="2076"></td>
        <td id="LC2076" class="blob-code blob-code-inner js-file-line">        VRMieComponent._gtable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2077" class="blob-num js-line-number" data-line-number="2077"></td>
        <td id="LC2077" class="blob-code blob-code-inner js-file-line">        VRMieComponent._gtable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2078" class="blob-num js-line-number" data-line-number="2078"></td>
        <td id="LC2078" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2079" class="blob-num js-line-number" data-line-number="2079"></td>
        <td id="LC2079" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> gij</td>
      </tr>
      <tr>
        <td id="L2080" class="blob-num js-line-number" data-line-number="2080"></td>
        <td id="LC2080" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2081" class="blob-num js-line-number" data-line-number="2081"></td>
        <td id="LC2081" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> class AssocSite(object):</span></td>
      </tr>
      <tr>
        <td id="L2082" class="blob-num js-line-number" data-line-number="2082"></td>
        <td id="LC2082" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>     _rc_table = np.array([[0.]])</span></td>
      </tr>
      <tr>
        <td id="L2083" class="blob-num js-line-number" data-line-number="2083"></td>
        <td id="LC2083" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>     def __init__(self):</span></td>
      </tr>
      <tr>
        <td id="L2084" class="blob-num js-line-number" data-line-number="2084"></td>
        <td id="LC2084" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2085" class="blob-num js-line-number" data-line-number="2085"></td>
        <td id="LC2085" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">GMieGroup</span>(<span class="pl-e">MieGroup</span>):</td>
      </tr>
      <tr>
        <td id="L2086" class="blob-num js-line-number" data-line-number="2086"></td>
        <td id="LC2086" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2087" class="blob-num js-line-number" data-line-number="2087"></td>
        <td id="LC2087" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    SAFT-gamma Mie Group as defined by SAFT-gamma Mie Equation-of-state</span></td>
      </tr>
      <tr>
        <td id="L2088" class="blob-num js-line-number" data-line-number="2088"></td>
        <td id="LC2088" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455</span></td>
      </tr>
      <tr>
        <td id="L2089" class="blob-num js-line-number" data-line-number="2089"></td>
        <td id="LC2089" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2090" class="blob-num js-line-number" data-line-number="2090"></td>
        <td id="LC2090" class="blob-code blob-code-inner js-file-line">    _total <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L2091" class="blob-num js-line-number" data-line-number="2091"></td>
        <td id="LC2091" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">lambda_r</span>, <span class="pl-smi">lambda_a</span>, <span class="pl-smi">sigma</span>, <span class="pl-smi">epsilon</span>, <span class="pl-smi">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">None</span>, <span class="pl-smi">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">1</span>., <span class="pl-smi">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-smi">name</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L2092" class="blob-num js-line-number" data-line-number="2092"></td>
        <td id="LC2092" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">super</span>().<span class="pl-c1">__init__</span>(lambda_r, lambda_a, sigma, epsilon)</td>
      </tr>
      <tr>
        <td id="L2093" class="blob-num js-line-number" data-line-number="2093"></td>
        <td id="LC2093" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.sf <span class="pl-k">=</span> shape_factor</td>
      </tr>
      <tr>
        <td id="L2094" class="blob-num js-line-number" data-line-number="2094"></td>
        <td id="LC2094" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.vk <span class="pl-k">=</span> id_seg</td>
      </tr>
      <tr>
        <td id="L2095" class="blob-num js-line-number" data-line-number="2095"></td>
        <td id="LC2095" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.mw <span class="pl-k">=</span> molar_weight</td>
      </tr>
      <tr>
        <td id="L2096" class="blob-num js-line-number" data-line-number="2096"></td>
        <td id="LC2096" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2097" class="blob-num js-line-number" data-line-number="2097"></td>
        <td id="LC2097" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.index <span class="pl-k">=</span> GMieGroup._total</td>
      </tr>
      <tr>
        <td id="L2098" class="blob-num js-line-number" data-line-number="2098"></td>
        <td id="LC2098" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.name <span class="pl-k">=</span> name</td>
      </tr>
      <tr>
        <td id="L2099" class="blob-num js-line-number" data-line-number="2099"></td>
        <td id="LC2099" class="blob-code blob-code-inner js-file-line">        GMieGroup._total <span class="pl-k">+=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L2100" class="blob-num js-line-number" data-line-number="2100"></td>
        <td id="LC2100" class="blob-code blob-code-inner js-file-line">        GMieGroup.add_elem()</td>
      </tr>
      <tr>
        <td id="L2101" class="blob-num js-line-number" data-line-number="2101"></td>
        <td id="LC2101" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2102" class="blob-num js-line-number" data-line-number="2102"></td>
        <td id="LC2102" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2103" class="blob-num js-line-number" data-line-number="2103"></td>
        <td id="LC2103" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2104" class="blob-num js-line-number" data-line-number="2104"></td>
        <td id="LC2104" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2105" class="blob-num js-line-number" data-line-number="2105"></td>
        <td id="LC2105" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__repr__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L2106" class="blob-num js-line-number" data-line-number="2106"></td>
        <td id="LC2106" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(<span class="pl-c1">self</span>.name, <span class="pl-c1">str</span>):</td>
      </tr>
      <tr>
        <td id="L2107" class="blob-num js-line-number" data-line-number="2107"></td>
        <td id="LC2107" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> <span class="pl-s"><span class="pl-pds">&quot;</span>G-Mie <span class="pl-c1">{<span class="pl-k">:s</span>}</span> group<span class="pl-pds">&quot;</span></span>.format(<span class="pl-c1">self</span>.name)</td>
      </tr>
      <tr>
        <td id="L2108" class="blob-num js-line-number" data-line-number="2108"></td>
        <td id="LC2108" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-s"><span class="pl-pds">&quot;</span>G-Mie: sf=<span class="pl-c1">{<span class="pl-k">:4.3f</span>}</span>, id_seg=<span class="pl-c1">{<span class="pl-k">:4.3f</span>}</span> <span class="pl-pds">&quot;</span></span>.format(<span class="pl-c1">self</span>.sf, <span class="pl-c1">self</span>.vk) <span class="pl-k">+</span> <span class="pl-c1">super</span>().<span class="pl-c1">__repr__</span>()</td>
      </tr>
      <tr>
        <td id="L2109" class="blob-num js-line-number" data-line-number="2109"></td>
        <td id="LC2109" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2110" class="blob-num js-line-number" data-line-number="2110"></td>
        <td id="LC2110" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cp_ideal</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">cpparam</span>, <span class="pl-smi">ref</span><span class="pl-k">=</span>(<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.,<span class="pl-c1">273.15</span>,<span class="pl-c1">101325</span>)):</td>
      </tr>
      <tr>
        <td id="L2111" class="blob-num js-line-number" data-line-number="2111"></td>
        <td id="LC2111" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(cpparam, (np.ndarray, <span class="pl-c1">list</span>, <span class="pl-c1">tuple</span>)) <span class="pl-k">and</span> <span class="pl-c1">len</span>(cpparam)<span class="pl-k">==</span><span class="pl-c1">4</span>,</td>
      </tr>
      <tr>
        <td id="L2112" class="blob-num js-line-number" data-line-number="2112"></td>
        <td id="LC2112" class="blob-code blob-code-inner js-file-line">            <span class="pl-s"><span class="pl-pds">&quot;</span>Cp parameters must be iterable with length of 4<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2113" class="blob-num js-line-number" data-line-number="2113"></td>
        <td id="LC2113" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Cp parameters in form [A, B, C, D]</span></td>
      </tr>
      <tr>
        <td id="L2114" class="blob-num js-line-number" data-line-number="2114"></td>
        <td id="LC2114" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> cpparam</td>
      </tr>
      <tr>
        <td id="L2115" class="blob-num js-line-number" data-line-number="2115"></td>
        <td id="LC2115" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> ref <span class="pl-c"><span class="pl-c">#</span> ref: h0, s0, T0, P0</span></td>
      </tr>
      <tr>
        <td id="L2116" class="blob-num js-line-number" data-line-number="2116"></td>
        <td id="LC2116" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2117" class="blob-num js-line-number" data-line-number="2117"></td>
        <td id="LC2117" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> Combining rules with exceptions build-in</span></td>
      </tr>
      <tr>
        <td id="L2118" class="blob-num js-line-number" data-line-number="2118"></td>
        <td id="LC2118" class="blob-code blob-code-inner js-file-line">    _ktable <span class="pl-k">=</span> np.array([[<span class="pl-c1">0</span>.]])</td>
      </tr>
      <tr>
        <td id="L2119" class="blob-num js-line-number" data-line-number="2119"></td>
        <td id="LC2119" class="blob-code blob-code-inner js-file-line">    _gtable <span class="pl-k">=</span> np.array([[<span class="pl-c1">0</span>.]])</td>
      </tr>
      <tr>
        <td id="L2120" class="blob-num js-line-number" data-line-number="2120"></td>
        <td id="LC2120" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2121" class="blob-num js-line-number" data-line-number="2121"></td>
        <td id="LC2121" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2122" class="blob-num js-line-number" data-line-number="2122"></td>
        <td id="LC2122" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">n_total</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L2123" class="blob-num js-line-number" data-line-number="2123"></td>
        <td id="LC2123" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">cls</span>._total</td>
      </tr>
      <tr>
        <td id="L2124" class="blob-num js-line-number" data-line-number="2124"></td>
        <td id="LC2124" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2125" class="blob-num js-line-number" data-line-number="2125"></td>
        <td id="LC2125" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2126" class="blob-num js-line-number" data-line-number="2126"></td>
        <td id="LC2126" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">add_elem</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L2127" class="blob-num js-line-number" data-line-number="2127"></td>
        <td id="LC2127" class="blob-code blob-code-inner js-file-line">        t1 <span class="pl-k">=</span> <span class="pl-c1">cls</span>._ktable</td>
      </tr>
      <tr>
        <td id="L2128" class="blob-num js-line-number" data-line-number="2128"></td>
        <td id="LC2128" class="blob-code blob-code-inner js-file-line">        t2 <span class="pl-k">=</span> <span class="pl-c1">cls</span>._gtable</td>
      </tr>
      <tr>
        <td id="L2129" class="blob-num js-line-number" data-line-number="2129"></td>
        <td id="LC2129" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">cls</span>._total <span class="pl-k">&gt;</span> <span class="pl-c1">1</span>:</td>
      </tr>
      <tr>
        <td id="L2130" class="blob-num js-line-number" data-line-number="2130"></td>
        <td id="LC2130" class="blob-code blob-code-inner js-file-line">            t12 <span class="pl-k">=</span> np.append(t1, np.zeros((<span class="pl-c1">1</span>, t1.shape[<span class="pl-c1">1</span>])), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L2131" class="blob-num js-line-number" data-line-number="2131"></td>
        <td id="LC2131" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">cls</span>._ktable <span class="pl-k">=</span> np.append(t12, np.zeros((t12.shape[<span class="pl-c1">0</span>], <span class="pl-c1">1</span>)), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L2132" class="blob-num js-line-number" data-line-number="2132"></td>
        <td id="LC2132" class="blob-code blob-code-inner js-file-line">            t22 <span class="pl-k">=</span> np.append(t2, np.zeros((<span class="pl-c1">1</span>, t2.shape[<span class="pl-c1">1</span>])), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">0</span>)</td>
      </tr>
      <tr>
        <td id="L2133" class="blob-num js-line-number" data-line-number="2133"></td>
        <td id="LC2133" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">cls</span>._gtable <span class="pl-k">=</span> np.append(t22, np.zeros((t22.shape[<span class="pl-c1">0</span>], <span class="pl-c1">1</span>)), <span class="pl-v">axis</span><span class="pl-k">=</span><span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L2134" class="blob-num js-line-number" data-line-number="2134"></td>
        <td id="LC2134" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2135" class="blob-num js-line-number" data-line-number="2135"></td>
        <td id="LC2135" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__add__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>):</td>
      </tr>
      <tr>
        <td id="L2136" class="blob-num js-line-number" data-line-number="2136"></td>
        <td id="LC2136" class="blob-code blob-code-inner js-file-line">        mg <span class="pl-k">=</span> <span class="pl-c1">super</span>().<span class="pl-c1">__add__</span>(other)</td>
      </tr>
      <tr>
        <td id="L2137" class="blob-num js-line-number" data-line-number="2137"></td>
        <td id="LC2137" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> other <span class="pl-k">is</span> <span class="pl-c1">self</span>:</td>
      </tr>
      <tr>
        <td id="L2138" class="blob-num js-line-number" data-line-number="2138"></td>
        <td id="LC2138" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> mg</td>
      </tr>
      <tr>
        <td id="L2139" class="blob-num js-line-number" data-line-number="2139"></td>
        <td id="LC2139" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> GMieGroup._ktable[<span class="pl-c1">self</span>.index, other.index]</td>
      </tr>
      <tr>
        <td id="L2140" class="blob-num js-line-number" data-line-number="2140"></td>
        <td id="LC2140" class="blob-code blob-code-inner js-file-line">        mg.epsilon <span class="pl-k">=</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> kij) <span class="pl-k">*</span> mg.epsilon</td>
      </tr>
      <tr>
        <td id="L2141" class="blob-num js-line-number" data-line-number="2141"></td>
        <td id="LC2141" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> GMieGroup._gtable[<span class="pl-c1">self</span>.index, other.index]</td>
      </tr>
      <tr>
        <td id="L2142" class="blob-num js-line-number" data-line-number="2142"></td>
        <td id="LC2142" class="blob-code blob-code-inner js-file-line">        mg.rep <span class="pl-k">=</span> <span class="pl-c1">3</span> <span class="pl-k">+</span> (<span class="pl-c1">1</span> <span class="pl-k">-</span> gij) <span class="pl-k">*</span> (mg.rep <span class="pl-k">-</span> <span class="pl-c1">3</span>)</td>
      </tr>
      <tr>
        <td id="L2143" class="blob-num js-line-number" data-line-number="2143"></td>
        <td id="LC2143" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> mg</td>
      </tr>
      <tr>
        <td id="L2144" class="blob-num js-line-number" data-line-number="2144"></td>
        <td id="LC2144" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2145" class="blob-num js-line-number" data-line-number="2145"></td>
        <td id="LC2145" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2146" class="blob-num js-line-number" data-line-number="2146"></td>
        <td id="LC2146" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">print_table</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L2147" class="blob-num js-line-number" data-line-number="2147"></td>
        <td id="LC2147" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:30s</span>}{<span class="pl-k">:d</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Total Gamma Mie Groups: <span class="pl-pds">&quot;</span></span>, <span class="pl-c1">cls</span>._total))</td>
      </tr>
      <tr>
        <td id="L2148" class="blob-num js-line-number" data-line-number="2148"></td>
        <td id="LC2148" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:40s</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Combination k values for Epsilon:<span class="pl-pds">&quot;</span></span>))</td>
      </tr>
      <tr>
        <td id="L2149" class="blob-num js-line-number" data-line-number="2149"></td>
        <td id="LC2149" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-c1">cls</span>._ktable)</td>
      </tr>
      <tr>
        <td id="L2150" class="blob-num js-line-number" data-line-number="2150"></td>
        <td id="LC2150" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-c1">{<span class="pl-k">:40s</span>}</span><span class="pl-pds">&quot;</span></span>.format(<span class="pl-s"><span class="pl-pds">&quot;</span>Combination gamma values for Lambda_r:<span class="pl-pds">&quot;</span></span>))</td>
      </tr>
      <tr>
        <td id="L2151" class="blob-num js-line-number" data-line-number="2151"></td>
        <td id="LC2151" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">print</span>(<span class="pl-c1">cls</span>._gtable)</td>
      </tr>
      <tr>
        <td id="L2152" class="blob-num js-line-number" data-line-number="2152"></td>
        <td id="LC2152" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2153" class="blob-num js-line-number" data-line-number="2153"></td>
        <td id="LC2153" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2154" class="blob-num js-line-number" data-line-number="2154"></td>
        <td id="LC2154" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_e_kij</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2155" class="blob-num js-line-number" data-line-number="2155"></td>
        <td id="LC2155" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, GMieGroup) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2156" class="blob-num js-line-number" data-line-number="2156"></td>
        <td id="LC2156" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2157" class="blob-num js-line-number" data-line-number="2157"></td>
        <td id="LC2157" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2158" class="blob-num js-line-number" data-line-number="2158"></td>
        <td id="LC2158" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2159" class="blob-num js-line-number" data-line-number="2159"></td>
        <td id="LC2159" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2160" class="blob-num js-line-number" data-line-number="2160"></td>
        <td id="LC2160" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2161" class="blob-num js-line-number" data-line-number="2161"></td>
        <td id="LC2161" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2162" class="blob-num js-line-number" data-line-number="2162"></td>
        <td id="LC2162" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_e_val</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2163" class="blob-num js-line-number" data-line-number="2163"></td>
        <td id="LC2163" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, GMieGroup) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2164" class="blob-num js-line-number" data-line-number="2164"></td>
        <td id="LC2164" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2165" class="blob-num js-line-number" data-line-number="2165"></td>
        <td id="LC2165" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2166" class="blob-num js-line-number" data-line-number="2166"></td>
        <td id="LC2166" class="blob-code blob-code-inner js-file-line">        s1 <span class="pl-k">=</span> g1.sigma</td>
      </tr>
      <tr>
        <td id="L2167" class="blob-num js-line-number" data-line-number="2167"></td>
        <td id="LC2167" class="blob-code blob-code-inner js-file-line">        s2 <span class="pl-k">=</span> g2.sigma</td>
      </tr>
      <tr>
        <td id="L2168" class="blob-num js-line-number" data-line-number="2168"></td>
        <td id="LC2168" class="blob-code blob-code-inner js-file-line">        e1 <span class="pl-k">=</span> g1.epsilon</td>
      </tr>
      <tr>
        <td id="L2169" class="blob-num js-line-number" data-line-number="2169"></td>
        <td id="LC2169" class="blob-code blob-code-inner js-file-line">        e2 <span class="pl-k">=</span> g2.epsilon</td>
      </tr>
      <tr>
        <td id="L2170" class="blob-num js-line-number" data-line-number="2170"></td>
        <td id="LC2170" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2171" class="blob-num js-line-number" data-line-number="2171"></td>
        <td id="LC2171" class="blob-code blob-code-inner js-file-line">        actl <span class="pl-k">=</span> sqrt(<span class="pl-c1">pow</span>(s1,<span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(s2,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>((s1<span class="pl-k">+</span>s2)<span class="pl-k">/</span><span class="pl-c1">2</span>, <span class="pl-c1">3</span>) <span class="pl-k">*</span> sqrt(e1 <span class="pl-k">*</span> e2)</td>
      </tr>
      <tr>
        <td id="L2172" class="blob-num js-line-number" data-line-number="2172"></td>
        <td id="LC2172" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> val <span class="pl-k">/</span> actl</td>
      </tr>
      <tr>
        <td id="L2173" class="blob-num js-line-number" data-line-number="2173"></td>
        <td id="LC2173" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2174" class="blob-num js-line-number" data-line-number="2174"></td>
        <td id="LC2174" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2175" class="blob-num js-line-number" data-line-number="2175"></td>
        <td id="LC2175" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2176" class="blob-num js-line-number" data-line-number="2176"></td>
        <td id="LC2176" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2177" class="blob-num js-line-number" data-line-number="2177"></td>
        <td id="LC2177" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2178" class="blob-num js-line-number" data-line-number="2178"></td>
        <td id="LC2178" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_lr_gij</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2179" class="blob-num js-line-number" data-line-number="2179"></td>
        <td id="LC2179" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, GMieGroup) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2180" class="blob-num js-line-number" data-line-number="2180"></td>
        <td id="LC2180" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2181" class="blob-num js-line-number" data-line-number="2181"></td>
        <td id="LC2181" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2182" class="blob-num js-line-number" data-line-number="2182"></td>
        <td id="LC2182" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2183" class="blob-num js-line-number" data-line-number="2183"></td>
        <td id="LC2183" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2184" class="blob-num js-line-number" data-line-number="2184"></td>
        <td id="LC2184" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2185" class="blob-num js-line-number" data-line-number="2185"></td>
        <td id="LC2185" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2186" class="blob-num js-line-number" data-line-number="2186"></td>
        <td id="LC2186" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">combining_lr_val</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>, <span class="pl-smi">g1</span>, <span class="pl-smi">g2</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2187" class="blob-num js-line-number" data-line-number="2187"></td>
        <td id="LC2187" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(g1, GMieGroup) <span class="pl-k">and</span> <span class="pl-c1">isinstance</span>(g2, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2188" class="blob-num js-line-number" data-line-number="2188"></td>
        <td id="LC2188" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2189" class="blob-num js-line-number" data-line-number="2189"></td>
        <td id="LC2189" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2190" class="blob-num js-line-number" data-line-number="2190"></td>
        <td id="LC2190" class="blob-code blob-code-inner js-file-line">        lr1 <span class="pl-k">=</span> g1.rep</td>
      </tr>
      <tr>
        <td id="L2191" class="blob-num js-line-number" data-line-number="2191"></td>
        <td id="LC2191" class="blob-code blob-code-inner js-file-line">        lr2 <span class="pl-k">=</span> g2.rep</td>
      </tr>
      <tr>
        <td id="L2192" class="blob-num js-line-number" data-line-number="2192"></td>
        <td id="LC2192" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2193" class="blob-num js-line-number" data-line-number="2193"></td>
        <td id="LC2193" class="blob-code blob-code-inner js-file-line">        intm <span class="pl-k">=</span> val <span class="pl-k">-</span> <span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L2194" class="blob-num js-line-number" data-line-number="2194"></td>
        <td id="LC2194" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> intm <span class="pl-k">/</span> sqrt((lr1 <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (lr2 <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2195" class="blob-num js-line-number" data-line-number="2195"></td>
        <td id="LC2195" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2196" class="blob-num js-line-number" data-line-number="2196"></td>
        <td id="LC2196" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i1,i2] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2197" class="blob-num js-line-number" data-line-number="2197"></td>
        <td id="LC2197" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">cls</span>._gtable[i2,i1] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2198" class="blob-num js-line-number" data-line-number="2198"></td>
        <td id="LC2198" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2199" class="blob-num js-line-number" data-line-number="2199"></td>
        <td id="LC2199" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_epsilon</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2200" class="blob-num js-line-number" data-line-number="2200"></td>
        <td id="LC2200" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2201" class="blob-num js-line-number" data-line-number="2201"></td>
        <td id="LC2201" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2202" class="blob-num js-line-number" data-line-number="2202"></td>
        <td id="LC2202" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2203" class="blob-num js-line-number" data-line-number="2203"></td>
        <td id="LC2203" class="blob-code blob-code-inner js-file-line">        s1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.sigma</td>
      </tr>
      <tr>
        <td id="L2204" class="blob-num js-line-number" data-line-number="2204"></td>
        <td id="LC2204" class="blob-code blob-code-inner js-file-line">        s2 <span class="pl-k">=</span> other.sigma</td>
      </tr>
      <tr>
        <td id="L2205" class="blob-num js-line-number" data-line-number="2205"></td>
        <td id="LC2205" class="blob-code blob-code-inner js-file-line">        e1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.epsilon</td>
      </tr>
      <tr>
        <td id="L2206" class="blob-num js-line-number" data-line-number="2206"></td>
        <td id="LC2206" class="blob-code blob-code-inner js-file-line">        e2 <span class="pl-k">=</span> other.epsilon</td>
      </tr>
      <tr>
        <td id="L2207" class="blob-num js-line-number" data-line-number="2207"></td>
        <td id="LC2207" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2208" class="blob-num js-line-number" data-line-number="2208"></td>
        <td id="LC2208" class="blob-code blob-code-inner js-file-line">        actl <span class="pl-k">=</span> sqrt(<span class="pl-c1">pow</span>(s1,<span class="pl-c1">3</span>) <span class="pl-k">*</span> <span class="pl-c1">pow</span>(s2,<span class="pl-c1">3</span>)) <span class="pl-k">/</span> <span class="pl-c1">pow</span>((s1<span class="pl-k">+</span>s2)<span class="pl-k">/</span><span class="pl-c1">2</span>, <span class="pl-c1">3</span>) <span class="pl-k">*</span> sqrt(e1 <span class="pl-k">*</span> e2)</td>
      </tr>
      <tr>
        <td id="L2209" class="blob-num js-line-number" data-line-number="2209"></td>
        <td id="LC2209" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> val <span class="pl-k">/</span> actl</td>
      </tr>
      <tr>
        <td id="L2210" class="blob-num js-line-number" data-line-number="2210"></td>
        <td id="LC2210" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2211" class="blob-num js-line-number" data-line-number="2211"></td>
        <td id="LC2211" class="blob-code blob-code-inner js-file-line">        GMieGroup._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2212" class="blob-num js-line-number" data-line-number="2212"></td>
        <td id="LC2212" class="blob-code blob-code-inner js-file-line">        GMieGroup._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2213" class="blob-num js-line-number" data-line-number="2213"></td>
        <td id="LC2213" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2214" class="blob-num js-line-number" data-line-number="2214"></td>
        <td id="LC2214" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> kij</td>
      </tr>
      <tr>
        <td id="L2215" class="blob-num js-line-number" data-line-number="2215"></td>
        <td id="LC2215" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2216" class="blob-num js-line-number" data-line-number="2216"></td>
        <td id="LC2216" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_e_kij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2217" class="blob-num js-line-number" data-line-number="2217"></td>
        <td id="LC2217" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2218" class="blob-num js-line-number" data-line-number="2218"></td>
        <td id="LC2218" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2219" class="blob-num js-line-number" data-line-number="2219"></td>
        <td id="LC2219" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2220" class="blob-num js-line-number" data-line-number="2220"></td>
        <td id="LC2220" class="blob-code blob-code-inner js-file-line">        kij <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2221" class="blob-num js-line-number" data-line-number="2221"></td>
        <td id="LC2221" class="blob-code blob-code-inner js-file-line">        GMieGroup._ktable[i1,i2] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2222" class="blob-num js-line-number" data-line-number="2222"></td>
        <td id="LC2222" class="blob-code blob-code-inner js-file-line">        GMieGroup._ktable[i2,i1] <span class="pl-k">=</span> kij</td>
      </tr>
      <tr>
        <td id="L2223" class="blob-num js-line-number" data-line-number="2223"></td>
        <td id="LC2223" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2224" class="blob-num js-line-number" data-line-number="2224"></td>
        <td id="LC2224" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> kij</td>
      </tr>
      <tr>
        <td id="L2225" class="blob-num js-line-number" data-line-number="2225"></td>
        <td id="LC2225" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2226" class="blob-num js-line-number" data-line-number="2226"></td>
        <td id="LC2226" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_rep</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2227" class="blob-num js-line-number" data-line-number="2227"></td>
        <td id="LC2227" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2228" class="blob-num js-line-number" data-line-number="2228"></td>
        <td id="LC2228" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2229" class="blob-num js-line-number" data-line-number="2229"></td>
        <td id="LC2229" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2230" class="blob-num js-line-number" data-line-number="2230"></td>
        <td id="LC2230" class="blob-code blob-code-inner js-file-line">        lr1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.rep</td>
      </tr>
      <tr>
        <td id="L2231" class="blob-num js-line-number" data-line-number="2231"></td>
        <td id="LC2231" class="blob-code blob-code-inner js-file-line">        lr2 <span class="pl-k">=</span> other.rep</td>
      </tr>
      <tr>
        <td id="L2232" class="blob-num js-line-number" data-line-number="2232"></td>
        <td id="LC2232" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2233" class="blob-num js-line-number" data-line-number="2233"></td>
        <td id="LC2233" class="blob-code blob-code-inner js-file-line">        intm <span class="pl-k">=</span> val <span class="pl-k">-</span> <span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L2234" class="blob-num js-line-number" data-line-number="2234"></td>
        <td id="LC2234" class="blob-code blob-code-inner js-file-line">        ratio <span class="pl-k">=</span> intm <span class="pl-k">/</span> sqrt((lr1 <span class="pl-k">-</span> <span class="pl-c1">3</span>) <span class="pl-k">*</span> (lr2 <span class="pl-k">-</span> <span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2235" class="blob-num js-line-number" data-line-number="2235"></td>
        <td id="LC2235" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> <span class="pl-c1">1</span> <span class="pl-k">-</span> ratio</td>
      </tr>
      <tr>
        <td id="L2236" class="blob-num js-line-number" data-line-number="2236"></td>
        <td id="LC2236" class="blob-code blob-code-inner js-file-line">        GMieGroup._gtable[i1,i2] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2237" class="blob-num js-line-number" data-line-number="2237"></td>
        <td id="LC2237" class="blob-code blob-code-inner js-file-line">        GMieGroup._gtable[i2,i1] <span class="pl-k">=</span> gij</td>
      </tr>
      <tr>
        <td id="L2238" class="blob-num js-line-number" data-line-number="2238"></td>
        <td id="LC2238" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2239" class="blob-num js-line-number" data-line-number="2239"></td>
        <td id="LC2239" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> gij</td>
      </tr>
      <tr>
        <td id="L2240" class="blob-num js-line-number" data-line-number="2240"></td>
        <td id="LC2240" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2241" class="blob-num js-line-number" data-line-number="2241"></td>
        <td id="LC2241" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">set_cross_r_gij</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">other</span>, <span class="pl-smi">val</span>):</td>
      </tr>
      <tr>
        <td id="L2242" class="blob-num js-line-number" data-line-number="2242"></td>
        <td id="LC2242" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(other, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Set cross interactions between same class only<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2243" class="blob-num js-line-number" data-line-number="2243"></td>
        <td id="LC2243" class="blob-code blob-code-inner js-file-line">        i1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.index</td>
      </tr>
      <tr>
        <td id="L2244" class="blob-num js-line-number" data-line-number="2244"></td>
        <td id="LC2244" class="blob-code blob-code-inner js-file-line">        i2 <span class="pl-k">=</span> other.index</td>
      </tr>
      <tr>
        <td id="L2245" class="blob-num js-line-number" data-line-number="2245"></td>
        <td id="LC2245" class="blob-code blob-code-inner js-file-line">        gij <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2246" class="blob-num js-line-number" data-line-number="2246"></td>
        <td id="LC2246" class="blob-code blob-code-inner js-file-line">        GMieGroup._gtable[i1,i2] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2247" class="blob-num js-line-number" data-line-number="2247"></td>
        <td id="LC2247" class="blob-code blob-code-inner js-file-line">        GMieGroup._gtable[i2,i1] <span class="pl-k">=</span> val</td>
      </tr>
      <tr>
        <td id="L2248" class="blob-num js-line-number" data-line-number="2248"></td>
        <td id="LC2248" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2249" class="blob-num js-line-number" data-line-number="2249"></td>
        <td id="LC2249" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> gij</td>
      </tr>
      <tr>
        <td id="L2250" class="blob-num js-line-number" data-line-number="2250"></td>
        <td id="LC2250" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2251" class="blob-num js-line-number" data-line-number="2251"></td>
        <td id="LC2251" class="blob-code blob-code-inner js-file-line"><span class="pl-k">class</span> <span class="pl-en">GMieComponent</span>(<span class="pl-e">Component</span>):</td>
      </tr>
      <tr>
        <td id="L2252" class="blob-num js-line-number" data-line-number="2252"></td>
        <td id="LC2252" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2253" class="blob-num js-line-number" data-line-number="2253"></td>
        <td id="LC2253" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    SAFT-gamma Mie Component as defined by SAFT-gamma Mie Equation-of-state</span></td>
      </tr>
      <tr>
        <td id="L2254" class="blob-num js-line-number" data-line-number="2254"></td>
        <td id="LC2254" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    given in ref: J. Chem. Phys. 140, 054107 (2014); https://doi.org/10.1063/1.4851455</span></td>
      </tr>
      <tr>
        <td id="L2255" class="blob-num js-line-number" data-line-number="2255"></td>
        <td id="LC2255" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2256" class="blob-num js-line-number" data-line-number="2256"></td>
        <td id="LC2256" class="blob-code blob-code-inner js-file-line">    _total <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L2257" class="blob-num js-line-number" data-line-number="2257"></td>
        <td id="LC2257" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__init__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L2258" class="blob-num js-line-number" data-line-number="2258"></td>
        <td id="LC2258" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> molar_weight <span class="pl-k">==</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2259" class="blob-num js-line-number" data-line-number="2259"></td>
        <td id="LC2259" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">super</span>().<span class="pl-c1">__init__</span>(<span class="pl-c1">0</span>.)</td>
      </tr>
      <tr>
        <td id="L2260" class="blob-num js-line-number" data-line-number="2260"></td>
        <td id="LC2260" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L2261" class="blob-num js-line-number" data-line-number="2261"></td>
        <td id="LC2261" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(molar_weight,(<span class="pl-c1">int</span>,<span class="pl-c1">float</span>)):</td>
      </tr>
      <tr>
        <td id="L2262" class="blob-num js-line-number" data-line-number="2262"></td>
        <td id="LC2262" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">super</span>().<span class="pl-c1">__init__</span>(molar_weight)</td>
      </tr>
      <tr>
        <td id="L2263" class="blob-num js-line-number" data-line-number="2263"></td>
        <td id="LC2263" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.groups <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L2264" class="blob-num js-line-number" data-line-number="2264"></td>
        <td id="LC2264" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ngtype <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L2265" class="blob-num js-line-number" data-line-number="2265"></td>
        <td id="LC2265" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ngroups <span class="pl-k">=</span> <span class="pl-c1">0</span></td>
      </tr>
      <tr>
        <td id="L2266" class="blob-num js-line-number" data-line-number="2266"></td>
        <td id="LC2266" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.index <span class="pl-k">=</span> GMieComponent._total</td>
      </tr>
      <tr>
        <td id="L2267" class="blob-num js-line-number" data-line-number="2267"></td>
        <td id="LC2267" class="blob-code blob-code-inner js-file-line">        GMieComponent._total <span class="pl-k">+=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L2268" class="blob-num js-line-number" data-line-number="2268"></td>
        <td id="LC2268" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2269" class="blob-num js-line-number" data-line-number="2269"></td>
        <td id="LC2269" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-c1">__repr__</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>):</td>
      </tr>
      <tr>
        <td id="L2270" class="blob-num js-line-number" data-line-number="2270"></td>
        <td id="LC2270" class="blob-code blob-code-inner js-file-line">        s <span class="pl-k">=</span> <span class="pl-s"><span class="pl-pds">&#39;</span>G-Mie Component with <span class="pl-c1">{<span class="pl-k">:d</span>}</span> groups<span class="pl-cce">\n</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-c1">self</span>.ngroups)</td>
      </tr>
      <tr>
        <td id="L2271" class="blob-num js-line-number" data-line-number="2271"></td>
        <td id="LC2271" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2272" class="blob-num js-line-number" data-line-number="2272"></td>
        <td id="LC2272" class="blob-code blob-code-inner js-file-line">            s <span class="pl-k">+=</span> <span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:d</span>}</span> x <span class="pl-c1">{<span class="pl-k">:s</span>}</span><span class="pl-cce">\n</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-c1">self</span>.groups[g], <span class="pl-c1">repr</span>(g))</td>
      </tr>
      <tr>
        <td id="L2273" class="blob-num js-line-number" data-line-number="2273"></td>
        <td id="LC2273" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> s</td>
      </tr>
      <tr>
        <td id="L2274" class="blob-num js-line-number" data-line-number="2274"></td>
        <td id="LC2274" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2275" class="blob-num js-line-number" data-line-number="2275"></td>
        <td id="LC2275" class="blob-code blob-code-inner js-file-line">    <span class="pl-en">@</span><span class="pl-c1">classmethod</span></td>
      </tr>
      <tr>
        <td id="L2276" class="blob-num js-line-number" data-line-number="2276"></td>
        <td id="LC2276" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">n_total</span>(<span class="pl-smi"><span class="pl-smi">cls</span></span>):</td>
      </tr>
      <tr>
        <td id="L2277" class="blob-num js-line-number" data-line-number="2277"></td>
        <td id="LC2277" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">cls</span>._total</td>
      </tr>
      <tr>
        <td id="L2278" class="blob-num js-line-number" data-line-number="2278"></td>
        <td id="LC2278" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2279" class="blob-num js-line-number" data-line-number="2279"></td>
        <td id="LC2279" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">quick_set</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-k">*</span><span class="pl-smi">args</span>):</td>
      </tr>
      <tr>
        <td id="L2280" class="blob-num js-line-number" data-line-number="2280"></td>
        <td id="LC2280" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> arg <span class="pl-k">in</span> args:</td>
      </tr>
      <tr>
        <td id="L2281" class="blob-num js-line-number" data-line-number="2281"></td>
        <td id="LC2281" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg, <span class="pl-c1">tuple</span>) <span class="pl-k">or</span> <span class="pl-c1">isinstance</span>(arg, <span class="pl-c1">list</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Use iterable tuple or list for quick_set<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2282" class="blob-num js-line-number" data-line-number="2282"></td>
        <td id="LC2282" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg[<span class="pl-c1">0</span>], GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>First element of iterable must be an instance of GMieGroup class<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2283" class="blob-num js-line-number" data-line-number="2283"></td>
        <td id="LC2283" class="blob-code blob-code-inner js-file-line">            mt.checkerr(<span class="pl-c1">isinstance</span>(arg[<span class="pl-c1">1</span>], <span class="pl-c1">int</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Second element of iterable must be of int type<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2284" class="blob-num js-line-number" data-line-number="2284"></td>
        <td id="LC2284" class="blob-code blob-code-inner js-file-line">            mt.checkerr(arg[<span class="pl-c1">1</span>] <span class="pl-k">&gt;</span> <span class="pl-c1">0</span>, <span class="pl-s"><span class="pl-pds">&quot;</span>Second element of iterable must be of integer &gt; 0<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2285" class="blob-num js-line-number" data-line-number="2285"></td>
        <td id="LC2285" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.groups[arg[<span class="pl-c1">0</span>]] <span class="pl-k">=</span> arg[<span class="pl-c1">1</span>]</td>
      </tr>
      <tr>
        <td id="L2286" class="blob-num js-line-number" data-line-number="2286"></td>
        <td id="LC2286" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2287" class="blob-num js-line-number" data-line-number="2287"></td>
        <td id="LC2287" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ngroups <span class="pl-k">=</span> <span class="pl-c1">sum</span>(<span class="pl-c1">self</span>.groups.values())</td>
      </tr>
      <tr>
        <td id="L2288" class="blob-num js-line-number" data-line-number="2288"></td>
        <td id="LC2288" class="blob-code blob-code-inner js-file-line">        <span class="pl-c1">self</span>.ngtype <span class="pl-k">=</span> <span class="pl-c1">len</span>(<span class="pl-c1">self</span>.groups)</td>
      </tr>
      <tr>
        <td id="L2289" class="blob-num js-line-number" data-line-number="2289"></td>
        <td id="LC2289" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2290" class="blob-num js-line-number" data-line-number="2290"></td>
        <td id="LC2290" class="blob-code blob-code-inner js-file-line">        <span class="pl-c"><span class="pl-c">#</span> Update molar weight</span></td>
      </tr>
      <tr>
        <td id="L2291" class="blob-num js-line-number" data-line-number="2291"></td>
        <td id="LC2291" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2292" class="blob-num js-line-number" data-line-number="2292"></td>
        <td id="LC2292" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">if</span> g.mw <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2293" class="blob-num js-line-number" data-line-number="2293"></td>
        <td id="LC2293" class="blob-code blob-code-inner js-file-line">                <span class="pl-c1">self</span>.mw <span class="pl-k">+=</span> g.mw <span class="pl-k">*</span> <span class="pl-c1">self</span>.groups[g]</td>
      </tr>
      <tr>
        <td id="L2294" class="blob-num js-line-number" data-line-number="2294"></td>
        <td id="LC2294" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2295" class="blob-num js-line-number" data-line-number="2295"></td>
        <td id="LC2295" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L2296" class="blob-num js-line-number" data-line-number="2296"></td>
        <td id="LC2296" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2297" class="blob-num js-line-number" data-line-number="2297"></td>
        <td id="LC2297" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">add_group</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">group</span>, <span class="pl-smi">number</span>):</td>
      </tr>
      <tr>
        <td id="L2298" class="blob-num js-line-number" data-line-number="2298"></td>
        <td id="LC2298" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(group, GMieGroup), <span class="pl-s"><span class="pl-pds">&quot;</span>Group entry must be an instance of GMieGroup class<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2299" class="blob-num js-line-number" data-line-number="2299"></td>
        <td id="LC2299" class="blob-code blob-code-inner js-file-line">        mt.checkerr(<span class="pl-c1">isinstance</span>(number, <span class="pl-c1">int</span>), <span class="pl-s"><span class="pl-pds">&quot;</span>Only integer amount of molecules can be added<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2300" class="blob-num js-line-number" data-line-number="2300"></td>
        <td id="LC2300" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2301" class="blob-num js-line-number" data-line-number="2301"></td>
        <td id="LC2301" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> group <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2302" class="blob-num js-line-number" data-line-number="2302"></td>
        <td id="LC2302" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.groups[group] <span class="pl-k">+=</span> number</td>
      </tr>
      <tr>
        <td id="L2303" class="blob-num js-line-number" data-line-number="2303"></td>
        <td id="LC2303" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ngroups <span class="pl-k">+=</span> number</td>
      </tr>
      <tr>
        <td id="L2304" class="blob-num js-line-number" data-line-number="2304"></td>
        <td id="LC2304" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">else</span>:</td>
      </tr>
      <tr>
        <td id="L2305" class="blob-num js-line-number" data-line-number="2305"></td>
        <td id="LC2305" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.groups[group] <span class="pl-k">=</span> number</td>
      </tr>
      <tr>
        <td id="L2306" class="blob-num js-line-number" data-line-number="2306"></td>
        <td id="LC2306" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ngroups <span class="pl-k">+=</span> number</td>
      </tr>
      <tr>
        <td id="L2307" class="blob-num js-line-number" data-line-number="2307"></td>
        <td id="LC2307" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ngtype <span class="pl-k">+=</span> <span class="pl-c1">1</span></td>
      </tr>
      <tr>
        <td id="L2308" class="blob-num js-line-number" data-line-number="2308"></td>
        <td id="LC2308" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2309" class="blob-num js-line-number" data-line-number="2309"></td>
        <td id="LC2309" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> group.mw <span class="pl-k">is</span> <span class="pl-k">not</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2310" class="blob-num js-line-number" data-line-number="2310"></td>
        <td id="LC2310" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.mw <span class="pl-k">+=</span> g.mw <span class="pl-k">*</span> number</td>
      </tr>
      <tr>
        <td id="L2311" class="blob-num js-line-number" data-line-number="2311"></td>
        <td id="LC2311" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2312" class="blob-num js-line-number" data-line-number="2312"></td>
        <td id="LC2312" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">self</span></td>
      </tr>
      <tr>
        <td id="L2313" class="blob-num js-line-number" data-line-number="2313"></td>
        <td id="LC2313" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2314" class="blob-num js-line-number" data-line-number="2314"></td>
        <td id="LC2314" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">get_gtypeii</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">hsd</span><span class="pl-k">=</span><span class="pl-c1">None</span>):</td>
      </tr>
      <tr>
        <td id="L2315" class="blob-num js-line-number" data-line-number="2315"></td>
        <td id="LC2315" class="blob-code blob-code-inner js-file-line">        sig3 <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2316" class="blob-num js-line-number" data-line-number="2316"></td>
        <td id="LC2316" class="blob-code blob-code-inner js-file-line">        epsi <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2317" class="blob-num js-line-number" data-line-number="2317"></td>
        <td id="LC2317" class="blob-code blob-code-inner js-file-line">        rep <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2318" class="blob-num js-line-number" data-line-number="2318"></td>
        <td id="LC2318" class="blob-code blob-code-inner js-file-line">        att <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2319" class="blob-num js-line-number" data-line-number="2319"></td>
        <td id="LC2319" class="blob-code blob-code-inner js-file-line">        hsdii <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2320" class="blob-num js-line-number" data-line-number="2320"></td>
        <td id="LC2320" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2321" class="blob-num js-line-number" data-line-number="2321"></td>
        <td id="LC2321" class="blob-code blob-code-inner js-file-line">        den <span class="pl-k">=</span> <span class="pl-c1">0</span>.</td>
      </tr>
      <tr>
        <td id="L2322" class="blob-num js-line-number" data-line-number="2322"></td>
        <td id="LC2322" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2323" class="blob-num js-line-number" data-line-number="2323"></td>
        <td id="LC2323" class="blob-code blob-code-inner js-file-line">            den <span class="pl-k">+=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.vk <span class="pl-k">*</span> g.sf</td>
      </tr>
      <tr>
        <td id="L2324" class="blob-num js-line-number" data-line-number="2324"></td>
        <td id="LC2324" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2325" class="blob-num js-line-number" data-line-number="2325"></td>
        <td id="LC2325" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">for</span> g1 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2326" class="blob-num js-line-number" data-line-number="2326"></td>
        <td id="LC2326" class="blob-code blob-code-inner js-file-line">            num1 <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g1] <span class="pl-k">*</span> g1.vk <span class="pl-k">*</span> g1.sf <span class="pl-c"><span class="pl-c">#</span> v_ki * v*_k * Sk</span></td>
      </tr>
      <tr>
        <td id="L2327" class="blob-num js-line-number" data-line-number="2327"></td>
        <td id="LC2327" class="blob-code blob-code-inner js-file-line">            zki <span class="pl-k">=</span> num1 <span class="pl-k">/</span> den</td>
      </tr>
      <tr>
        <td id="L2328" class="blob-num js-line-number" data-line-number="2328"></td>
        <td id="LC2328" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g2 <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups: </td>
      </tr>
      <tr>
        <td id="L2329" class="blob-num js-line-number" data-line-number="2329"></td>
        <td id="LC2329" class="blob-code blob-code-inner js-file-line">                num2 <span class="pl-k">=</span> <span class="pl-c1">self</span>.groups[g2] <span class="pl-k">*</span> g2.vk <span class="pl-k">*</span> g2.sf <span class="pl-c"><span class="pl-c">#</span> v_ki * v*_k * Sk</span></td>
      </tr>
      <tr>
        <td id="L2330" class="blob-num js-line-number" data-line-number="2330"></td>
        <td id="LC2330" class="blob-code blob-code-inner js-file-line">                zli <span class="pl-k">=</span> num2 <span class="pl-k">/</span> den</td>
      </tr>
      <tr>
        <td id="L2331" class="blob-num js-line-number" data-line-number="2331"></td>
        <td id="LC2331" class="blob-code blob-code-inner js-file-line">                gc <span class="pl-k">=</span> g1<span class="pl-k">+</span>g2</td>
      </tr>
      <tr>
        <td id="L2332" class="blob-num js-line-number" data-line-number="2332"></td>
        <td id="LC2332" class="blob-code blob-code-inner js-file-line">                sig3 <span class="pl-k">+=</span> zki <span class="pl-k">*</span> zli <span class="pl-k">*</span> gc.sigma<span class="pl-k">**</span><span class="pl-c1">3</span></td>
      </tr>
      <tr>
        <td id="L2333" class="blob-num js-line-number" data-line-number="2333"></td>
        <td id="LC2333" class="blob-code blob-code-inner js-file-line">                epsi <span class="pl-k">+=</span> zki <span class="pl-k">*</span> zli <span class="pl-k">*</span> gc.epsilon</td>
      </tr>
      <tr>
        <td id="L2334" class="blob-num js-line-number" data-line-number="2334"></td>
        <td id="LC2334" class="blob-code blob-code-inner js-file-line">                rep <span class="pl-k">+=</span> zki <span class="pl-k">*</span> zli <span class="pl-k">*</span> gc.rep</td>
      </tr>
      <tr>
        <td id="L2335" class="blob-num js-line-number" data-line-number="2335"></td>
        <td id="LC2335" class="blob-code blob-code-inner js-file-line">                att <span class="pl-k">+=</span> zki <span class="pl-k">*</span> zli <span class="pl-k">*</span> gc.att</td>
      </tr>
      <tr>
        <td id="L2336" class="blob-num js-line-number" data-line-number="2336"></td>
        <td id="LC2336" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> <span class="pl-c1">isinstance</span>(hsd, np.ndarray):</td>
      </tr>
      <tr>
        <td id="L2337" class="blob-num js-line-number" data-line-number="2337"></td>
        <td id="LC2337" class="blob-code blob-code-inner js-file-line">                    g1i <span class="pl-k">=</span> g1.index</td>
      </tr>
      <tr>
        <td id="L2338" class="blob-num js-line-number" data-line-number="2338"></td>
        <td id="LC2338" class="blob-code blob-code-inner js-file-line">                    g2i <span class="pl-k">=</span> g2.index</td>
      </tr>
      <tr>
        <td id="L2339" class="blob-num js-line-number" data-line-number="2339"></td>
        <td id="LC2339" class="blob-code blob-code-inner js-file-line">                    hsdii <span class="pl-k">+=</span> zki <span class="pl-k">*</span> zli <span class="pl-k">*</span> hsd[g1i, g2i]</td>
      </tr>
      <tr>
        <td id="L2340" class="blob-num js-line-number" data-line-number="2340"></td>
        <td id="LC2340" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2341" class="blob-num js-line-number" data-line-number="2341"></td>
        <td id="LC2341" class="blob-code blob-code-inner js-file-line">        mg <span class="pl-k">=</span> MieGroup(rep, att, sig3<span class="pl-k">**</span>(<span class="pl-c1">1</span><span class="pl-k">/</span><span class="pl-c1">3</span>), epsi)</td>
      </tr>
      <tr>
        <td id="L2342" class="blob-num js-line-number" data-line-number="2342"></td>
        <td id="LC2342" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2343" class="blob-num js-line-number" data-line-number="2343"></td>
        <td id="LC2343" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-k">not</span> <span class="pl-c1">isinstance</span>(hsd, np.ndarray):</td>
      </tr>
      <tr>
        <td id="L2344" class="blob-num js-line-number" data-line-number="2344"></td>
        <td id="LC2344" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">return</span> mg</td>
      </tr>
      <tr>
        <td id="L2345" class="blob-num js-line-number" data-line-number="2345"></td>
        <td id="LC2345" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> (mg, hsdii)</td>
      </tr>
      <tr>
        <td id="L2346" class="blob-num js-line-number" data-line-number="2346"></td>
        <td id="LC2346" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2347" class="blob-num js-line-number" data-line-number="2347"></td>
        <td id="LC2347" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">cp_int</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">T</span>):</td>
      </tr>
      <tr>
        <td id="L2348" class="blob-num js-line-number" data-line-number="2348"></td>
        <td id="LC2348" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2349" class="blob-num js-line-number" data-line-number="2349"></td>
        <td id="LC2349" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> [<span class="pl-c1">0</span>] <span class="pl-k">*</span> <span class="pl-c1">4</span></td>
      </tr>
      <tr>
        <td id="L2350" class="blob-num js-line-number" data-line-number="2350"></td>
        <td id="LC2350" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2351" class="blob-num js-line-number" data-line-number="2351"></td>
        <td id="LC2351" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> g.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2352" class="blob-num js-line-number" data-line-number="2352"></td>
        <td id="LC2352" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2353" class="blob-num js-line-number" data-line-number="2353"></td>
        <td id="LC2353" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">return</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2354" class="blob-num js-line-number" data-line-number="2354"></td>
        <td id="LC2354" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-c1">4</span>):</td>
      </tr>
      <tr>
        <td id="L2355" class="blob-num js-line-number" data-line-number="2355"></td>
        <td id="LC2355" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.cp[i] <span class="pl-k">+=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.cp[i]</td>
      </tr>
      <tr>
        <td id="L2356" class="blob-num js-line-number" data-line-number="2356"></td>
        <td id="LC2356" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> (<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.,<span class="pl-c1">273.15</span>,<span class="pl-c1">101325</span>) <span class="pl-c"><span class="pl-c">#</span> default reference state</span></td>
      </tr>
      <tr>
        <td id="L2357" class="blob-num js-line-number" data-line-number="2357"></td>
        <td id="LC2357" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2358" class="blob-num js-line-number" data-line-number="2358"></td>
        <td id="LC2358" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">super</span>().cp_int(T)</td>
      </tr>
      <tr>
        <td id="L2359" class="blob-num js-line-number" data-line-number="2359"></td>
        <td id="LC2359" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2360" class="blob-num js-line-number" data-line-number="2360"></td>
        <td id="LC2360" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">def</span> <span class="pl-en">cp_t_int</span>(<span class="pl-smi"><span class="pl-smi">self</span></span>, <span class="pl-smi">T</span>):</td>
      </tr>
      <tr>
        <td id="L2361" class="blob-num js-line-number" data-line-number="2361"></td>
        <td id="LC2361" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">if</span> <span class="pl-c1">self</span>.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2362" class="blob-num js-line-number" data-line-number="2362"></td>
        <td id="LC2362" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> [<span class="pl-c1">0</span>] <span class="pl-k">*</span> <span class="pl-c1">4</span></td>
      </tr>
      <tr>
        <td id="L2363" class="blob-num js-line-number" data-line-number="2363"></td>
        <td id="LC2363" class="blob-code blob-code-inner js-file-line">            <span class="pl-k">for</span> g <span class="pl-k">in</span> <span class="pl-c1">self</span>.groups:</td>
      </tr>
      <tr>
        <td id="L2364" class="blob-num js-line-number" data-line-number="2364"></td>
        <td id="LC2364" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">if</span> g.cp <span class="pl-k">is</span> <span class="pl-c1">None</span>:</td>
      </tr>
      <tr>
        <td id="L2365" class="blob-num js-line-number" data-line-number="2365"></td>
        <td id="LC2365" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.cp <span class="pl-k">=</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2366" class="blob-num js-line-number" data-line-number="2366"></td>
        <td id="LC2366" class="blob-code blob-code-inner js-file-line">                    <span class="pl-k">return</span> <span class="pl-c1">None</span></td>
      </tr>
      <tr>
        <td id="L2367" class="blob-num js-line-number" data-line-number="2367"></td>
        <td id="LC2367" class="blob-code blob-code-inner js-file-line">                <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-c1">4</span>):</td>
      </tr>
      <tr>
        <td id="L2368" class="blob-num js-line-number" data-line-number="2368"></td>
        <td id="LC2368" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c1">self</span>.cp[i] <span class="pl-k">+=</span> <span class="pl-c1">self</span>.groups[g] <span class="pl-k">*</span> g.cp[i]</td>
      </tr>
      <tr>
        <td id="L2369" class="blob-num js-line-number" data-line-number="2369"></td>
        <td id="LC2369" class="blob-code blob-code-inner js-file-line">            <span class="pl-c1">self</span>.ref <span class="pl-k">=</span> (<span class="pl-c1">0</span>.,<span class="pl-c1">0</span>.,<span class="pl-c1">273.15</span>,<span class="pl-c1">101325</span>) <span class="pl-c"><span class="pl-c">#</span> default reference state</span></td>
      </tr>
      <tr>
        <td id="L2370" class="blob-num js-line-number" data-line-number="2370"></td>
        <td id="LC2370" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2371" class="blob-num js-line-number" data-line-number="2371"></td>
        <td id="LC2371" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">return</span> <span class="pl-c1">super</span>().cp_t_int(T)</td>
      </tr>
      <tr>
        <td id="L2372" class="blob-num js-line-number" data-line-number="2372"></td>
        <td id="LC2372" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2373" class="blob-num js-line-number" data-line-number="2373"></td>
        <td id="LC2373" class="blob-code blob-code-inner js-file-line"><span class="pl-k">def</span> <span class="pl-en">main</span>():</td>
      </tr>
      <tr>
        <td id="L2374" class="blob-num js-line-number" data-line-number="2374"></td>
        <td id="LC2374" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH4</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">12.504</span>, <span class="pl-c1">6</span>., <span class="pl-c1">3.737</span>, <span class="pl-c1">152.575</span>)</td>
      </tr>
      <tr>
        <td id="L2375" class="blob-num js-line-number" data-line-number="2375"></td>
        <td id="LC2375" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH3</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">15.04982</span>, <span class="pl-c1">6</span>., <span class="pl-c1">4.077257</span>, <span class="pl-c1">256.7662</span>)</td>
      </tr>
      <tr>
        <td id="L2376" class="blob-num js-line-number" data-line-number="2376"></td>
        <td id="LC2376" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH2</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">19.87107</span>, <span class="pl-c1">6</span>., <span class="pl-c1">4.880081</span>, <span class="pl-c1">473.3893</span>)</td>
      </tr>
      <tr>
        <td id="L2377" class="blob-num js-line-number" data-line-number="2377"></td>
        <td id="LC2377" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">COO</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">31.189</span>, <span class="pl-c1">6</span>., <span class="pl-c1">3.9939</span>, <span class="pl-c1">868.92</span>)</td>
      </tr>
      <tr>
        <td id="L2378" class="blob-num js-line-number" data-line-number="2378"></td>
        <td id="LC2378" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CO2</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">26.408</span>, <span class="pl-c1">5.055</span>, <span class="pl-c1">3.05</span>, <span class="pl-c1">207.891</span>)</td>
      </tr>
      <tr>
        <td id="L2379" class="blob-num js-line-number" data-line-number="2379"></td>
        <td id="LC2379" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">8.0</span>, <span class="pl-c1">6.0</span>, <span class="pl-c1">5.295</span>, <span class="pl-c1">95.621</span>)</td>
      </tr>
      <tr>
        <td id="L2380" class="blob-num js-line-number" data-line-number="2380"></td>
        <td id="LC2380" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">C2H6</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">10.16</span>, <span class="pl-c1">6</span>., <span class="pl-c1">3.488</span>, <span class="pl-c1">165.513</span>)</td>
      </tr>
      <tr>
        <td id="L2381" class="blob-num js-line-number" data-line-number="2381"></td>
        <td id="LC2381" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">LJ</span> <span class="pl-k">=</span> MieGroup(<span class="pl-c1">12</span>,<span class="pl-c1">6</span>,<span class="pl-c1">1</span>,<span class="pl-c1">1</span>)</td>
      </tr>
      <tr>
        <td id="L2382" class="blob-num js-line-number" data-line-number="2382"></td>
        <td id="LC2382" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2383" class="blob-num js-line-number" data-line-number="2383"></td>
        <td id="LC2383" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2384" class="blob-num js-line-number" data-line-number="2384"></td>
        <td id="LC2384" class="blob-code blob-code-inner js-file-line">    vrc <span class="pl-k">=</span> {}</td>
      </tr>
      <tr>
        <td id="L2385" class="blob-num js-line-number" data-line-number="2385"></td>
        <td id="LC2385" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2386" class="blob-num js-line-number" data-line-number="2386"></td>
        <td id="LC2386" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> J. Chem. Phys. 139, 154504 (2013); https://doi.org/10.1063/1.4819786</span></td>
      </tr>
      <tr>
        <td id="L2387" class="blob-num js-line-number" data-line-number="2387"></td>
        <td id="LC2387" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>methane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1.0000</span>,  <span class="pl-c1">16.04</span>, <span class="pl-c1">12.650</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">3.7412</span>, <span class="pl-c1">153.36</span>)</td>
      </tr>
      <tr>
        <td id="L2388" class="blob-num js-line-number" data-line-number="2388"></td>
        <td id="LC2388" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>ethane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                 VRMieComponent(<span class="pl-c1">1.4373</span>,  <span class="pl-c1">30.07</span>, <span class="pl-c1">12.400</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">3.7257</span>, <span class="pl-c1">206.12</span>)</td>
      </tr>
      <tr>
        <td id="L2389" class="blob-num js-line-number" data-line-number="2389"></td>
        <td id="LC2389" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>propane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1.6845</span>,  <span class="pl-c1">44.10</span>, <span class="pl-c1">13.006</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">3.9056</span>, <span class="pl-c1">239.89</span>)</td>
      </tr>
      <tr>
        <td id="L2390" class="blob-num js-line-number" data-line-number="2390"></td>
        <td id="LC2390" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-butane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">1.8514</span>,  <span class="pl-c1">58.12</span>, <span class="pl-c1">13.650</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.0887</span>, <span class="pl-c1">273.64</span>)</td>
      </tr>
      <tr>
        <td id="L2391" class="blob-num js-line-number" data-line-number="2391"></td>
        <td id="LC2391" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-pentane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>              VRMieComponent(<span class="pl-c1">1.9606</span>,  <span class="pl-c1">72.15</span>, <span class="pl-c1">15.847</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.2928</span>, <span class="pl-c1">321.94</span>)</td>
      </tr>
      <tr>
        <td id="L2392" class="blob-num js-line-number" data-line-number="2392"></td>
        <td id="LC2392" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-hexane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">2.1097</span>,  <span class="pl-c1">86.18</span>, <span class="pl-c1">17.203</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.4230</span>, <span class="pl-c1">354.38</span>)</td>
      </tr>
      <tr>
        <td id="L2393" class="blob-num js-line-number" data-line-number="2393"></td>
        <td id="LC2393" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-heptane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>              VRMieComponent(<span class="pl-c1">2.3949</span>, <span class="pl-c1">100.21</span>, <span class="pl-c1">17.092</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.4282</span>, <span class="pl-c1">358.51</span>)</td>
      </tr>
      <tr>
        <td id="L2394" class="blob-num js-line-number" data-line-number="2394"></td>
        <td id="LC2394" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-octane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">2.6253</span>, <span class="pl-c1">114.23</span>, <span class="pl-c1">17.378</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.4696</span>, <span class="pl-c1">369.18</span>)</td>
      </tr>
      <tr>
        <td id="L2395" class="blob-num js-line-number" data-line-number="2395"></td>
        <td id="LC2395" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-nonane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">2.8099</span>, <span class="pl-c1">128.20</span>, <span class="pl-c1">18.324</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.5334</span>, <span class="pl-c1">387.55</span>)</td>
      </tr>
      <tr>
        <td id="L2396" class="blob-num js-line-number" data-line-number="2396"></td>
        <td id="LC2396" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">2.9976</span>, <span class="pl-c1">142.29</span>, <span class="pl-c1">18.885</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.5890</span>, <span class="pl-c1">400.79</span>)</td>
      </tr>
      <tr>
        <td id="L2397" class="blob-num js-line-number" data-line-number="2397"></td>
        <td id="LC2397" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-dodecane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>             VRMieComponent(<span class="pl-c1">3.2519</span>, <span class="pl-c1">170.33</span>, <span class="pl-c1">20.862</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.7484</span>, <span class="pl-c1">437.72</span>)</td>
      </tr>
      <tr>
        <td id="L2398" class="blob-num js-line-number" data-line-number="2398"></td>
        <td id="LC2398" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-pentadecane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>          VRMieComponent(<span class="pl-c1">3.9325</span>, <span class="pl-c1">212.41</span>, <span class="pl-c1">20.822</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.7738</span>, <span class="pl-c1">444.51</span>)</td>
      </tr>
      <tr>
        <td id="L2399" class="blob-num js-line-number" data-line-number="2399"></td>
        <td id="LC2399" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-eicosane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>             VRMieComponent(<span class="pl-c1">4.8794</span>, <span class="pl-c1">282.55</span>, <span class="pl-c1">22.926</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.8788</span>, <span class="pl-c1">475.76</span>)</td>
      </tr>
      <tr>
        <td id="L2400" class="blob-num js-line-number" data-line-number="2400"></td>
        <td id="LC2400" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>perfluoromethane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>       VRMieComponent(<span class="pl-c1">1.0000</span>,  <span class="pl-c1">88.00</span>, <span class="pl-c1">42.553</span>, <span class="pl-c1">5.1906</span>, <span class="pl-c1">4.3372</span>, <span class="pl-c1">232.62</span>)</td>
      </tr>
      <tr>
        <td id="L2401" class="blob-num js-line-number" data-line-number="2401"></td>
        <td id="LC2401" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>perfluoroethane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>        VRMieComponent(<span class="pl-c1">1.8529</span>, <span class="pl-c1">138.01</span>, <span class="pl-c1">19.192</span>, <span class="pl-c1">5.7506</span>, <span class="pl-c1">3.9336</span>, <span class="pl-c1">211.46</span>)</td>
      </tr>
      <tr>
        <td id="L2402" class="blob-num js-line-number" data-line-number="2402"></td>
        <td id="LC2402" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>perfluoropropane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>       VRMieComponent(<span class="pl-c1">1.9401</span>, <span class="pl-c1">188.02</span>, <span class="pl-c1">22.627</span>, <span class="pl-c1">5.7506</span>, <span class="pl-c1">4.2983</span>, <span class="pl-c1">263.26</span>)</td>
      </tr>
      <tr>
        <td id="L2403" class="blob-num js-line-number" data-line-number="2403"></td>
        <td id="LC2403" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-perfluorobutane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>      VRMieComponent(<span class="pl-c1">2.1983</span>, <span class="pl-c1">238.03</span>, <span class="pl-c1">24.761</span>, <span class="pl-c1">5.7506</span>, <span class="pl-c1">4.4495</span>, <span class="pl-c1">290.49</span>)</td>
      </tr>
      <tr>
        <td id="L2404" class="blob-num js-line-number" data-line-number="2404"></td>
        <td id="LC2404" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-perfluoropentane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>     VRMieComponent(<span class="pl-c1">2.3783</span>, <span class="pl-c1">288.03</span>, <span class="pl-c1">29.750</span>, <span class="pl-c1">5.7506</span>, <span class="pl-c1">4.6132</span>, <span class="pl-c1">328.56</span>)</td>
      </tr>
      <tr>
        <td id="L2405" class="blob-num js-line-number" data-line-number="2405"></td>
        <td id="LC2405" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-perfluorohexane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>      VRMieComponent(<span class="pl-c1">2.5202</span>, <span class="pl-c1">338.04</span>, <span class="pl-c1">30.741</span>, <span class="pl-c1">5.7506</span>, <span class="pl-c1">4.7885</span>, <span class="pl-c1">349.30</span>)</td>
      </tr>
      <tr>
        <td id="L2406" class="blob-num js-line-number" data-line-number="2406"></td>
        <td id="LC2406" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>fluorine<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>               VRMieComponent(<span class="pl-c1">1.3211</span>,  <span class="pl-c1">38.00</span>, <span class="pl-c1">11.606</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">2.9554</span>, <span class="pl-c1">96.268</span>)</td>
      </tr>
      <tr>
        <td id="L2407" class="blob-num js-line-number" data-line-number="2407"></td>
        <td id="LC2407" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>carbon dioxide<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>         VRMieComponent(<span class="pl-c1">1.5000</span>,  <span class="pl-c1">44.01</span>, <span class="pl-c1">27.557</span>, <span class="pl-c1">5.1646</span>, <span class="pl-c1">3.1916</span>, <span class="pl-c1">231.88</span>)</td>
      </tr>
      <tr>
        <td id="L2408" class="blob-num js-line-number" data-line-number="2408"></td>
        <td id="LC2408" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>benzene<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1.9163</span>,  <span class="pl-c1">78.11</span>, <span class="pl-c1">14.798</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.0549</span>, <span class="pl-c1">372.59</span>)</td>
      </tr>
      <tr>
        <td id="L2409" class="blob-num js-line-number" data-line-number="2409"></td>
        <td id="LC2409" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>toluene<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1.9977</span>,  <span class="pl-c1">92.14</span>, <span class="pl-c1">16.334</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.2777</span>, <span class="pl-c1">409.73</span>)</td>
      </tr>
      <tr>
        <td id="L2410" class="blob-num js-line-number" data-line-number="2410"></td>
        <td id="LC2410" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>ethane<span class="pl-pds">&#39;</span></span>].set_cross_e_kij(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>], <span class="pl-k">-</span><span class="pl-c1">0.0222</span>)</td>
      </tr>
      <tr>
        <td id="L2411" class="blob-num js-line-number" data-line-number="2411"></td>
        <td id="LC2411" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>carbon dioxide<span class="pl-pds">&#39;</span></span>].set_cross_e_kij(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">0.0500</span>)</td>
      </tr>
      <tr>
        <td id="L2412" class="blob-num js-line-number" data-line-number="2412"></td>
        <td id="LC2412" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2413" class="blob-num js-line-number" data-line-number="2413"></td>
        <td id="LC2413" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> J. Phys. Chem. B 2013, 117, 2717−2733</span></td>
      </tr>
      <tr>
        <td id="L2414" class="blob-num js-line-number" data-line-number="2414"></td>
        <td id="LC2414" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>CO2<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1</span>,  <span class="pl-c1">44.01</span>, <span class="pl-c1">23.000</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">3.7410</span>, <span class="pl-c1">353.55</span>)</td>
      </tr>
      <tr>
        <td id="L2415" class="blob-num js-line-number" data-line-number="2415"></td>
        <td id="LC2415" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>CF4<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1</span>,  <span class="pl-c1">88.00</span>, <span class="pl-c1">32.530</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">4.3500</span>, <span class="pl-c1">265.57</span>)</td>
      </tr>
      <tr>
        <td id="L2416" class="blob-num js-line-number" data-line-number="2416"></td>
        <td id="LC2416" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>SF6<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>                VRMieComponent(<span class="pl-c1">1</span>, <span class="pl-c1">146.06</span>, <span class="pl-c1">19.020</span>,   <span class="pl-c1">8.80</span>, <span class="pl-c1">4.8610</span>, <span class="pl-c1">440.49</span>)</td>
      </tr>
      <tr>
        <td id="L2417" class="blob-num js-line-number" data-line-number="2417"></td>
        <td id="LC2417" class="blob-code blob-code-inner js-file-line">    vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>HFO-1234yf<span class="pl-pds">&#39;</span></span>] <span class="pl-k">=</span>         VRMieComponent(<span class="pl-c1">2</span>, <span class="pl-c1">114.00</span>, <span class="pl-c1">16.540</span>,      <span class="pl-c1">6</span>, <span class="pl-c1">3.9000</span>, <span class="pl-c1">257.39</span>)</td>
      </tr>
      <tr>
        <td id="L2418" class="blob-num js-line-number" data-line-number="2418"></td>
        <td id="LC2418" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> VRMieComponent.print_table()</span></td>
      </tr>
      <tr>
        <td id="L2419" class="blob-num js-line-number" data-line-number="2419"></td>
        <td id="LC2419" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>ethane<span class="pl-pds">&#39;</span></span>] <span class="pl-k">+</span> vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>])</td>
      </tr>
      <tr>
        <td id="L2420" class="blob-num js-line-number" data-line-number="2420"></td>
        <td id="LC2420" class="blob-code blob-code-inner js-file-line">    s <span class="pl-k">=</span> SAFTVRSystem().quick_set((vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>methane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">800</span>), (vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">100</span>))</td>
      </tr>
      <tr>
        <td id="L2421" class="blob-num js-line-number" data-line-number="2421"></td>
        <td id="LC2421" class="blob-code blob-code-inner js-file-line">    s.add_comp(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>n-decane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">100</span>)</td>
      </tr>
      <tr>
        <td id="L2422" class="blob-num js-line-number" data-line-number="2422"></td>
        <td id="LC2422" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2423" class="blob-num js-line-number" data-line-number="2423"></td>
        <td id="LC2423" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> s = SAFTVRSystem().quick_set((vrc[&#39;methane&#39;], 800))</span></td>
      </tr>
      <tr>
        <td id="L2424" class="blob-num js-line-number" data-line-number="2424"></td>
        <td id="LC2424" class="blob-code blob-code-inner js-file-line">    s2 <span class="pl-k">=</span> SAFTVRSystem().quick_set((vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>propane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1000</span>),(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>ethane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1000</span>),(vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>methane<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1000</span>))<span class="pl-c"><span class="pl-c">#</span>,(vrc[&#39;n-butane&#39;], 1000),(vrc[&#39;n-pentane&#39;], 1000),(vrc[&#39;n-hexane&#39;], 1000),(vrc[&#39;n-decane&#39;], 1000))</span></td>
      </tr>
      <tr>
        <td id="L2425" class="blob-num js-line-number" data-line-number="2425"></td>
        <td id="LC2425" class="blob-code blob-code-inner js-file-line">    s2.volume <span class="pl-k">=</span> <span class="pl-c1">3000</span></td>
      </tr>
      <tr>
        <td id="L2426" class="blob-num js-line-number" data-line-number="2426"></td>
        <td id="LC2426" class="blob-code blob-code-inner js-file-line">    s.volume <span class="pl-k">=</span> Var(<span class="pl-c1">3000</span>)</td>
      </tr>
      <tr>
        <td id="L2427" class="blob-num js-line-number" data-line-number="2427"></td>
        <td id="LC2427" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> s.helmholtz()</span></td>
      </tr>
      <tr>
        <td id="L2428" class="blob-num js-line-number" data-line-number="2428"></td>
        <td id="LC2428" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;Testing VR Mie System ======&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2429" class="blob-num js-line-number" data-line-number="2429"></td>
        <td id="LC2429" class="blob-code blob-code-inner js-file-line">    sys.setrecursionlimit(<span class="pl-c1">3000</span>)</td>
      </tr>
      <tr>
        <td id="L2430" class="blob-num js-line-number" data-line-number="2430"></td>
        <td id="LC2430" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(sys.getrecursionlimit())</td>
      </tr>
      <tr>
        <td id="L2431" class="blob-num js-line-number" data-line-number="2431"></td>
        <td id="LC2431" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;xi_1&#39;),s._SAFTVRSystem__xi_m(1))</span></td>
      </tr>
      <tr>
        <td id="L2432" class="blob-num js-line-number" data-line-number="2432"></td>
        <td id="LC2432" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;xi_2&#39;),s._SAFTVRSystem__xi_m(2))</span></td>
      </tr>
      <tr>
        <td id="L2433" class="blob-num js-line-number" data-line-number="2433"></td>
        <td id="LC2433" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;xi_3&#39;),s._SAFTVRSystem__xi_m(3))</span></td>
      </tr>
      <tr>
        <td id="L2434" class="blob-num js-line-number" data-line-number="2434"></td>
        <td id="LC2434" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_hs&#39;),s.a_hs())</span></td>
      </tr>
      <tr>
        <td id="L2435" class="blob-num js-line-number" data-line-number="2435"></td>
        <td id="LC2435" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;xi_x&#39;),s._SAFTVRSystem__xi_x())</span></td>
      </tr>
      <tr>
        <td id="L2436" class="blob-num js-line-number" data-line-number="2436"></td>
        <td id="LC2436" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_1&#39;),s.a_1())</span></td>
      </tr>
      <tr>
        <td id="L2437" class="blob-num js-line-number" data-line-number="2437"></td>
        <td id="LC2437" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;xi_x&#39;),s._SAFTVRSystem__xi_sx())</span></td>
      </tr>
      <tr>
        <td id="L2438" class="blob-num js-line-number" data-line-number="2438"></td>
        <td id="LC2438" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_2&#39;),s.a_2())</span></td>
      </tr>
      <tr>
        <td id="L2439" class="blob-num js-line-number" data-line-number="2439"></td>
        <td id="LC2439" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_3&#39;),s.a_3())</span></td>
      </tr>
      <tr>
        <td id="L2440" class="blob-num js-line-number" data-line-number="2440"></td>
        <td id="LC2440" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_mono&#39;),s.a_mono())</span></td>
      </tr>
      <tr>
        <td id="L2441" class="blob-num js-line-number" data-line-number="2441"></td>
        <td id="LC2441" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;gdhs&#39;),s._SAFTVRSystem__gdhs(vrc[&#39;methane&#39;].index, vrc[&#39;methane&#39;].index))</span></td>
      </tr>
      <tr>
        <td id="L2442" class="blob-num js-line-number" data-line-number="2442"></td>
        <td id="LC2442" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;der_xix&#39;),s._SAFTVRSystem__der_xi_x())</span></td>
      </tr>
      <tr>
        <td id="L2443" class="blob-num js-line-number" data-line-number="2443"></td>
        <td id="LC2443" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;g1&#39;),s._SAFTVRSystem__g1(vrc[&#39;methane&#39;].index, vrc[&#39;methane&#39;].index))</span></td>
      </tr>
      <tr>
        <td id="L2444" class="blob-num js-line-number" data-line-number="2444"></td>
        <td id="LC2444" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;g2&#39;),s._SAFTVRSystem__g2(vrc[&#39;methane&#39;].index, vrc[&#39;methane&#39;].index))</span></td>
      </tr>
      <tr>
        <td id="L2445" class="blob-num js-line-number" data-line-number="2445"></td>
        <td id="LC2445" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;gmieij&#39;),s._SAFTVRSystem__gmieij(vrc[&#39;methane&#39;].index,vrc[&#39;methane&#39;].index))</span></td>
      </tr>
      <tr>
        <td id="L2446" class="blob-num js-line-number" data-line-number="2446"></td>
        <td id="LC2446" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;a_chain&#39;),s.a_chain())</span></td>
      </tr>
      <tr>
        <td id="L2447" class="blob-num js-line-number" data-line-number="2447"></td>
        <td id="LC2447" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;helmholtz&#39;),s.helmholtz())</span></td>
      </tr>
      <tr>
        <td id="L2448" class="blob-num js-line-number" data-line-number="2448"></td>
        <td id="LC2448" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;d a_ideal dV&#39;),derivative(s.helmholtz_ideal(), s.volume))</span></td>
      </tr>
      <tr>
        <td id="L2449" class="blob-num js-line-number" data-line-number="2449"></td>
        <td id="LC2449" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;d a_res dV&#39;),derivative(s.helmholtz_residual(), s.volume))</span></td>
      </tr>
      <tr>
        <td id="L2450" class="blob-num js-line-number" data-line-number="2450"></td>
        <td id="LC2450" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2451" class="blob-num js-line-number" data-line-number="2451"></td>
        <td id="LC2451" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;Testing System get_properties ======&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2452" class="blob-num js-line-number" data-line-number="2452"></td>
        <td id="LC2452" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;Pressure (Pa):&#39;), s.get_pressure(temperature=319, molar_volume=0.00014))</span></td>
      </tr>
      <tr>
        <td id="L2453" class="blob-num js-line-number" data-line-number="2453"></td>
        <td id="LC2453" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;Cv (J/(mol K)):&#39;), s.get_cv(temperature=319, molar_volume=0.00014))</span></td>
      </tr>
      <tr>
        <td id="L2454" class="blob-num js-line-number" data-line-number="2454"></td>
        <td id="LC2454" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2455" class="blob-num js-line-number" data-line-number="2455"></td>
        <td id="LC2455" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH3</span> <span class="pl-k">=</span>  GMieGroup(<span class="pl-c1">15.04982</span>,    <span class="pl-c1">6</span>., <span class="pl-c1">4.077257</span>, <span class="pl-c1">256.7662</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">15.03502</span>, <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.5725512</span>, <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>CH3<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2456" class="blob-num js-line-number" data-line-number="2456"></td>
        <td id="LC2456" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH2</span> <span class="pl-k">=</span>  GMieGroup(<span class="pl-c1">19.87107</span>,    <span class="pl-c1">6</span>., <span class="pl-c1">4.880081</span>, <span class="pl-c1">473.3893</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">14.02708</span>, <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.2293202</span>, <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>CH2<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2457" class="blob-num js-line-number" data-line-number="2457"></td>
        <td id="LC2457" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">COO</span> <span class="pl-k">=</span>  GMieGroup(  <span class="pl-c1">31.189</span>,    <span class="pl-c1">6</span>.,   <span class="pl-c1">3.9939</span>,   <span class="pl-c1">868.92</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">44.01</span>,    <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.65264</span>,   <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>COO<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2458" class="blob-num js-line-number" data-line-number="2458"></td>
        <td id="LC2458" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CO2</span> <span class="pl-k">=</span>  GMieGroup(  <span class="pl-c1">26.408</span>, <span class="pl-c1">5.055</span>,     <span class="pl-c1">3.05</span>,  <span class="pl-c1">207.891</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">44.01</span>,    <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.847</span>,     <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">2</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>CO2<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2459" class="blob-num js-line-number" data-line-number="2459"></td>
        <td id="LC2459" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH</span> <span class="pl-k">=</span>   GMieGroup(      <span class="pl-c1">8</span>.,    <span class="pl-c1">6</span>.,    <span class="pl-c1">5.295</span>,   <span class="pl-c1">95.621</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">13.01864</span>, <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.0721</span>,    <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>CH<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2460" class="blob-num js-line-number" data-line-number="2460"></td>
        <td id="LC2460" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">C2H6</span> <span class="pl-k">=</span> GMieGroup(   <span class="pl-c1">10.16</span>,    <span class="pl-c1">6</span>.,    <span class="pl-c1">3.488</span>,  <span class="pl-c1">165.513</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">30.07004</span>, <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">0.855</span>,     <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">2</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>C2H6<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2461" class="blob-num js-line-number" data-line-number="2461"></td>
        <td id="LC2461" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH4</span> <span class="pl-k">=</span>  GMieGroup(  <span class="pl-c1">12.504</span>,    <span class="pl-c1">6</span>.,    <span class="pl-c1">3.737</span>,  <span class="pl-c1">152.575</span>, <span class="pl-v">molar_weight</span><span class="pl-k">=</span><span class="pl-c1">16.04296</span>, <span class="pl-v">shape_factor</span><span class="pl-k">=</span><span class="pl-c1">1</span>,         <span class="pl-v">id_seg</span><span class="pl-k">=</span><span class="pl-c1">1</span>, <span class="pl-v">name</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&quot;</span>CH4<span class="pl-pds">&quot;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2462" class="blob-num js-line-number" data-line-number="2462"></td>
        <td id="LC2462" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2463" class="blob-num js-line-number" data-line-number="2463"></td>
        <td id="LC2463" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH3</span>.set_cp_ideal((<span class="pl-c1">19.5</span>,<span class="pl-k">-</span><span class="pl-c1">8.08e-3</span>,<span class="pl-c1">1.53e-4</span>,<span class="pl-k">-</span><span class="pl-c1">9.67e-8</span>))</td>
      </tr>
      <tr>
        <td id="L2464" class="blob-num js-line-number" data-line-number="2464"></td>
        <td id="LC2464" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">CH2</span>.set_cp_ideal((<span class="pl-k">-</span><span class="pl-c1">0.909</span>,<span class="pl-c1">0.095</span>,<span class="pl-k">-</span><span class="pl-c1">5.44e-5</span>,<span class="pl-c1">1.19e-8</span>))</td>
      </tr>
      <tr>
        <td id="L2465" class="blob-num js-line-number" data-line-number="2465"></td>
        <td id="LC2465" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">C2H6</span>.set_cp_ideal((<span class="pl-c1">12.4755767</span>,<span class="pl-c1">1.44286617e-1</span>,<span class="pl-k">-</span><span class="pl-c1">2.3523953e-5</span>,<span class="pl-k">-</span><span class="pl-c1">8.99334179e-9</span>))</td>
      </tr>
      <tr>
        <td id="L2466" class="blob-num js-line-number" data-line-number="2466"></td>
        <td id="LC2466" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2467" class="blob-num js-line-number" data-line-number="2467"></td>
        <td id="LC2467" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH3</span>, <span class="pl-c1">CH2</span>, <span class="pl-c1">350.77</span>)</td>
      </tr>
      <tr>
        <td id="L2468" class="blob-num js-line-number" data-line-number="2468"></td>
        <td id="LC2468" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH3</span>, <span class="pl-c1">COO</span>, <span class="pl-c1">402.75</span>)</td>
      </tr>
      <tr>
        <td id="L2469" class="blob-num js-line-number" data-line-number="2469"></td>
        <td id="LC2469" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">COO</span>, <span class="pl-c1">CH2</span>, <span class="pl-c1">498.86</span>)</td>
      </tr>
      <tr>
        <td id="L2470" class="blob-num js-line-number" data-line-number="2470"></td>
        <td id="LC2470" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH3</span>, <span class="pl-c1">CO2</span>, <span class="pl-c1">205.698</span>)</td>
      </tr>
      <tr>
        <td id="L2471" class="blob-num js-line-number" data-line-number="2471"></td>
        <td id="LC2471" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH3</span>, <span class="pl-c1">CH</span>, <span class="pl-c1">387.48</span>)</td>
      </tr>
      <tr>
        <td id="L2472" class="blob-num js-line-number" data-line-number="2472"></td>
        <td id="LC2472" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH2</span>, <span class="pl-c1">CO2</span>, <span class="pl-c1">276.453</span>)</td>
      </tr>
      <tr>
        <td id="L2473" class="blob-num js-line-number" data-line-number="2473"></td>
        <td id="LC2473" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH2</span>, <span class="pl-c1">CH</span>, <span class="pl-c1">506.21</span>)</td>
      </tr>
      <tr>
        <td id="L2474" class="blob-num js-line-number" data-line-number="2474"></td>
        <td id="LC2474" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">C2H6</span>, <span class="pl-c1">CO2</span>, <span class="pl-c1">175.751</span>)</td>
      </tr>
      <tr>
        <td id="L2475" class="blob-num js-line-number" data-line-number="2475"></td>
        <td id="LC2475" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2476" class="blob-num js-line-number" data-line-number="2476"></td>
        <td id="LC2476" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH3</span>, <span class="pl-c1">193.97079</span>)</td>
      </tr>
      <tr>
        <td id="L2477" class="blob-num js-line-number" data-line-number="2477"></td>
        <td id="LC2477" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_lr_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH3</span>, <span class="pl-c1">12.62762</span>)</td>
      </tr>
      <tr>
        <td id="L2478" class="blob-num js-line-number" data-line-number="2478"></td>
        <td id="LC2478" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH2</span>, <span class="pl-c1">243.12915</span>)</td>
      </tr>
      <tr>
        <td id="L2479" class="blob-num js-line-number" data-line-number="2479"></td>
        <td id="LC2479" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_lr_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH2</span>, <span class="pl-c1">12.64155</span>)</td>
      </tr>
      <tr>
        <td id="L2480" class="blob-num js-line-number" data-line-number="2480"></td>
        <td id="LC2480" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH</span>, <span class="pl-c1">297.2062</span>)</td>
      </tr>
      <tr>
        <td id="L2481" class="blob-num js-line-number" data-line-number="2481"></td>
        <td id="LC2481" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_lr_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CH</span>, <span class="pl-c1">10.996</span>)</td>
      </tr>
      <tr>
        <td id="L2482" class="blob-num js-line-number" data-line-number="2482"></td>
        <td id="LC2482" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_e_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CO2</span>, <span class="pl-c1">144.722</span>)</td>
      </tr>
      <tr>
        <td id="L2483" class="blob-num js-line-number" data-line-number="2483"></td>
        <td id="LC2483" class="blob-code blob-code-inner js-file-line">    GMieGroup.combining_lr_val(<span class="pl-c1">CH4</span>, <span class="pl-c1">CO2</span>, <span class="pl-c1">11.95</span>)</td>
      </tr>
      <tr>
        <td id="L2484" class="blob-num js-line-number" data-line-number="2484"></td>
        <td id="LC2484" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2485" class="blob-num js-line-number" data-line-number="2485"></td>
        <td id="LC2485" class="blob-code blob-code-inner js-file-line">    c <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L2486" class="blob-num js-line-number" data-line-number="2486"></td>
        <td id="LC2486" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(c.mw, <span class="pl-c1">3</span><span class="pl-k">*</span><span class="pl-c1">15.03502</span> <span class="pl-k">+</span> <span class="pl-c1">2</span><span class="pl-k">*</span><span class="pl-c1">14.02708</span>)</td>
      </tr>
      <tr>
        <td id="L2487" class="blob-num js-line-number" data-line-number="2487"></td>
        <td id="LC2487" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(c.get_gtypeii())</td>
      </tr>
      <tr>
        <td id="L2488" class="blob-num js-line-number" data-line-number="2488"></td>
        <td id="LC2488" class="blob-code blob-code-inner js-file-line">    c2 <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L2489" class="blob-num js-line-number" data-line-number="2489"></td>
        <td id="LC2489" class="blob-code blob-code-inner js-file-line">    c3 <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2490" class="blob-num js-line-number" data-line-number="2490"></td>
        <td id="LC2490" class="blob-code blob-code-inner js-file-line">    c4 <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">4</span>))</td>
      </tr>
      <tr>
        <td id="L2491" class="blob-num js-line-number" data-line-number="2491"></td>
        <td id="LC2491" class="blob-code blob-code-inner js-file-line">    c1 <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">C2H6</span>,<span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L2492" class="blob-num js-line-number" data-line-number="2492"></td>
        <td id="LC2492" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2493" class="blob-num js-line-number" data-line-number="2493"></td>
        <td id="LC2493" class="blob-code blob-code-inner js-file-line">    s <span class="pl-k">=</span> SAFTgMieSystem().quick_set((c1, <span class="pl-c1">1000</span>), (c4, <span class="pl-c1">500</span>))</td>
      </tr>
      <tr>
        <td id="L2494" class="blob-num js-line-number" data-line-number="2494"></td>
        <td id="LC2494" class="blob-code blob-code-inner js-file-line">    s.volume <span class="pl-k">=</span> <span class="pl-c1">3000</span></td>
      </tr>
      <tr>
        <td id="L2495" class="blob-num js-line-number" data-line-number="2495"></td>
        <td id="LC2495" class="blob-code blob-code-inner js-file-line">    s.helmholtz()</td>
      </tr>
      <tr>
        <td id="L2496" class="blob-num js-line-number" data-line-number="2496"></td>
        <td id="LC2496" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2497" class="blob-num js-line-number" data-line-number="2497"></td>
        <td id="LC2497" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>Testing Gamma Mie System ======<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2498" class="blob-num js-line-number" data-line-number="2498"></td>
        <td id="LC2498" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(s.groups)</td>
      </tr>
      <tr>
        <td id="L2499" class="blob-num js-line-number" data-line-number="2499"></td>
        <td id="LC2499" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>=<span class="pl-pds">&#39;</span></span><span class="pl-k">*</span><span class="pl-c1">30</span>)</td>
      </tr>
      <tr>
        <td id="L2500" class="blob-num js-line-number" data-line-number="2500"></td>
        <td id="LC2500" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>seg_ratio<span class="pl-pds">&#39;</span></span>),s.segratio)</td>
      </tr>
      <tr>
        <td id="L2501" class="blob-num js-line-number" data-line-number="2501"></td>
        <td id="LC2501" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>segden<span class="pl-pds">&#39;</span></span>),s.segden)</td>
      </tr>
      <tr>
        <td id="L2502" class="blob-num js-line-number" data-line-number="2502"></td>
        <td id="LC2502" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_ideal<span class="pl-pds">&#39;</span></span>),s.a_ideal())</td>
      </tr>
      <tr>
        <td id="L2503" class="blob-num js-line-number" data-line-number="2503"></td>
        <td id="LC2503" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_0<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_m(<span class="pl-c1">0</span>))</td>
      </tr>
      <tr>
        <td id="L2504" class="blob-num js-line-number" data-line-number="2504"></td>
        <td id="LC2504" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_1<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_m(<span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L2505" class="blob-num js-line-number" data-line-number="2505"></td>
        <td id="LC2505" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_2<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_m(<span class="pl-c1">2</span>))</td>
      </tr>
      <tr>
        <td id="L2506" class="blob-num js-line-number" data-line-number="2506"></td>
        <td id="LC2506" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_3<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_m(<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2507" class="blob-num js-line-number" data-line-number="2507"></td>
        <td id="LC2507" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_hs<span class="pl-pds">&#39;</span></span>),s.a_hs())</td>
      </tr>
      <tr>
        <td id="L2508" class="blob-num js-line-number" data-line-number="2508"></td>
        <td id="LC2508" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_x<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_x())</td>
      </tr>
      <tr>
        <td id="L2509" class="blob-num js-line-number" data-line-number="2509"></td>
        <td id="LC2509" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_1kl: <span class="pl-pds">&#39;</span></span>), s._SAFTgMieSystem__a_1kl(<span class="pl-c1">C2H6</span>.index, <span class="pl-c1">C2H6</span>.index))</td>
      </tr>
      <tr>
        <td id="L2510" class="blob-num js-line-number" data-line-number="2510"></td>
        <td id="LC2510" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_1<span class="pl-pds">&#39;</span></span>),s.a_1())</td>
      </tr>
      <tr>
        <td id="L2511" class="blob-num js-line-number" data-line-number="2511"></td>
        <td id="LC2511" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>xi_sx<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__xi_sx())</td>
      </tr>
      <tr>
        <td id="L2512" class="blob-num js-line-number" data-line-number="2512"></td>
        <td id="LC2512" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_2<span class="pl-pds">&#39;</span></span>),s.a_2())</td>
      </tr>
      <tr>
        <td id="L2513" class="blob-num js-line-number" data-line-number="2513"></td>
        <td id="LC2513" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_3<span class="pl-pds">&#39;</span></span>),s.a_3())</td>
      </tr>
      <tr>
        <td id="L2514" class="blob-num js-line-number" data-line-number="2514"></td>
        <td id="LC2514" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_mono<span class="pl-pds">&#39;</span></span>),s.a_mono())</td>
      </tr>
      <tr>
        <td id="L2515" class="blob-num js-line-number" data-line-number="2515"></td>
        <td id="LC2515" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>gdhs<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__gdhs(c1.index))</td>
      </tr>
      <tr>
        <td id="L2516" class="blob-num js-line-number" data-line-number="2516"></td>
        <td id="LC2516" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>gdhs<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__gdhs(c4.index))</td>
      </tr>
      <tr>
        <td id="L2517" class="blob-num js-line-number" data-line-number="2517"></td>
        <td id="LC2517" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>der_xix<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__der_xi_x())</td>
      </tr>
      <tr>
        <td id="L2518" class="blob-num js-line-number" data-line-number="2518"></td>
        <td id="LC2518" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(s.ccomb)</span></td>
      </tr>
      <tr>
        <td id="L2519" class="blob-num js-line-number" data-line-number="2519"></td>
        <td id="LC2519" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>g1<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__g1(c4.index))</td>
      </tr>
      <tr>
        <td id="L2520" class="blob-num js-line-number" data-line-number="2520"></td>
        <td id="LC2520" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>g2<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__g2(c4.index))</td>
      </tr>
      <tr>
        <td id="L2521" class="blob-num js-line-number" data-line-number="2521"></td>
        <td id="LC2521" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>gmieij<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__gmieij(c1.index))</td>
      </tr>
      <tr>
        <td id="L2522" class="blob-num js-line-number" data-line-number="2522"></td>
        <td id="LC2522" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>gmieij<span class="pl-pds">&#39;</span></span>),s._SAFTgMieSystem__gmieij(c4.index))</td>
      </tr>
      <tr>
        <td id="L2523" class="blob-num js-line-number" data-line-number="2523"></td>
        <td id="LC2523" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>a_chain<span class="pl-pds">&#39;</span></span>),s.a_chain())</td>
      </tr>
      <tr>
        <td id="L2524" class="blob-num js-line-number" data-line-number="2524"></td>
        <td id="LC2524" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:18s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>helmholtz<span class="pl-pds">&#39;</span></span>),s.helmholtz())</td>
      </tr>
      <tr>
        <td id="L2525" class="blob-num js-line-number" data-line-number="2525"></td>
        <td id="LC2525" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;d a_ideal dV&#39;),derivative(s.helmholtz_ideal(), s.volume))</span></td>
      </tr>
      <tr>
        <td id="L2526" class="blob-num js-line-number" data-line-number="2526"></td>
        <td id="LC2526" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(&#39;{:18s}&#39;.format(&#39;d a_res dV&#39;),derivative(s.helmholtz_residual(), s.volume))</span></td>
      </tr>
      <tr>
        <td id="L2527" class="blob-num js-line-number" data-line-number="2527"></td>
        <td id="LC2527" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>=<span class="pl-pds">&#39;</span></span><span class="pl-k">*</span><span class="pl-c1">20</span>)</td>
      </tr>
      <tr>
        <td id="L2528" class="blob-num js-line-number" data-line-number="2528"></td>
        <td id="LC2528" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>Testing get properties<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2529" class="blob-num js-line-number" data-line-number="2529"></td>
        <td id="LC2529" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>=<span class="pl-pds">&#39;</span></span><span class="pl-k">*</span><span class="pl-c1">20</span>)</td>
      </tr>
      <tr>
        <td id="L2530" class="blob-num js-line-number" data-line-number="2530"></td>
        <td id="LC2530" class="blob-code blob-code-inner js-file-line">    s <span class="pl-k">=</span> SAFTVRSystem().quick_set((vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>SF6<span class="pl-pds">&#39;</span></span>], <span class="pl-c1">1000</span>))</td>
      </tr>
      <tr>
        <td id="L2531" class="blob-num js-line-number" data-line-number="2531"></td>
        <td id="LC2531" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> co2 = GMieComponent().quick_set((CO2, 1))</span></td>
      </tr>
      <tr>
        <td id="L2532" class="blob-num js-line-number" data-line-number="2532"></td>
        <td id="LC2532" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> s = SAFTgMieSystem().quick_set((co2, 1000))</span></td>
      </tr>
      <tr>
        <td id="L2533" class="blob-num js-line-number" data-line-number="2533"></td>
        <td id="LC2533" class="blob-code blob-code-inner js-file-line">    (pc, tc, rhoc) <span class="pl-k">=</span> s.critical_point(<span class="pl-v">initial_t</span><span class="pl-k">=</span><span class="pl-c1">325.9</span>, <span class="pl-v">v_nd</span><span class="pl-k">=</span>np.logspace(<span class="pl-k">-</span><span class="pl-c1">4</span>,<span class="pl-k">-</span><span class="pl-c1">1</span>,<span class="pl-c1">70</span>), <span class="pl-v">get_volume</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">get_density</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">print_progress</span><span class="pl-k">=</span><span class="pl-c1">True</span>)</td>
      </tr>
      <tr>
        <td id="L2534" class="blob-num js-line-number" data-line-number="2534"></td>
        <td id="LC2534" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>critical P (bar)<span class="pl-pds">&#39;</span></span>, pc<span class="pl-k">*</span>cst.patobar))</td>
      </tr>
      <tr>
        <td id="L2535" class="blob-num js-line-number" data-line-number="2535"></td>
        <td id="LC2535" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>critical T (K)<span class="pl-pds">&#39;</span></span>, tc))</td>
      </tr>
      <tr>
        <td id="L2536" class="blob-num js-line-number" data-line-number="2536"></td>
        <td id="LC2536" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>critical rho (mol/m3)<span class="pl-pds">&#39;</span></span>, rhoc<span class="pl-k">*</span><span class="pl-c1">1e-3</span><span class="pl-k">*</span>vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>SF6<span class="pl-pds">&#39;</span></span>].mw))</td>
      </tr>
      <tr>
        <td id="L2537" class="blob-num js-line-number" data-line-number="2537"></td>
        <td id="LC2537" class="blob-code blob-code-inner js-file-line">    vget <span class="pl-k">=</span> s.single_phase_v(<span class="pl-c1">15e6</span>, <span class="pl-c1">550</span>, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">v_crit</span><span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">/</span>rhoc, <span class="pl-v">supercritical</span><span class="pl-k">=</span><span class="pl-c1">True</span>)</td>
      </tr>
      <tr>
        <td id="L2538" class="blob-num js-line-number" data-line-number="2538"></td>
        <td id="LC2538" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>sp_v (1e-3 m3/mol)<span class="pl-pds">&#39;</span></span>, vget <span class="pl-k">*</span> <span class="pl-c1">1e3</span>))</td>
      </tr>
      <tr>
        <td id="L2539" class="blob-num js-line-number" data-line-number="2539"></td>
        <td id="LC2539" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>rho (kg/m3)<span class="pl-pds">&#39;</span></span>,<span class="pl-c1">1</span><span class="pl-k">/</span>vget<span class="pl-k">*</span><span class="pl-c1">1e-3</span><span class="pl-k">*</span>vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>SF6<span class="pl-pds">&#39;</span></span>].mw))</td>
      </tr>
      <tr>
        <td id="L2540" class="blob-num js-line-number" data-line-number="2540"></td>
        <td id="LC2540" class="blob-code blob-code-inner js-file-line">    s.set_molar_volume(vget)</td>
      </tr>
      <tr>
        <td id="L2541" class="blob-num js-line-number" data-line-number="2541"></td>
        <td id="LC2541" class="blob-code blob-code-inner js-file-line">    s.temperature <span class="pl-k">=</span> <span class="pl-c1">550</span></td>
      </tr>
      <tr>
        <td id="L2542" class="blob-num js-line-number" data-line-number="2542"></td>
        <td id="LC2542" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_pressure (MPa)<span class="pl-pds">&#39;</span></span>,s.get_pressure()<span class="pl-k">*</span><span class="pl-c1">1e-6</span>))</td>
      </tr>
      <tr>
        <td id="L2543" class="blob-num js-line-number" data-line-number="2543"></td>
        <td id="LC2543" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_entropy (J/mol K)<span class="pl-pds">&#39;</span></span>,s.get_entropy()))</td>
      </tr>
      <tr>
        <td id="L2544" class="blob-num js-line-number" data-line-number="2544"></td>
        <td id="LC2544" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_u (J/mol)<span class="pl-pds">&#39;</span></span>,s.get_u()))</td>
      </tr>
      <tr>
        <td id="L2545" class="blob-num js-line-number" data-line-number="2545"></td>
        <td id="LC2545" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_enthalpy (J/mol)<span class="pl-pds">&#39;</span></span>,s.get_enthalpy()))</td>
      </tr>
      <tr>
        <td id="L2546" class="blob-num js-line-number" data-line-number="2546"></td>
        <td id="LC2546" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_gibbs (J/mol)<span class="pl-pds">&#39;</span></span>,s.get_gibbs()))</td>
      </tr>
      <tr>
        <td id="L2547" class="blob-num js-line-number" data-line-number="2547"></td>
        <td id="LC2547" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_cv (J/mol K)<span class="pl-pds">&#39;</span></span>,s.get_cv()))</td>
      </tr>
      <tr>
        <td id="L2548" class="blob-num js-line-number" data-line-number="2548"></td>
        <td id="LC2548" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_kappa (1/MPa)<span class="pl-pds">&#39;</span></span>,s.get_kappa() <span class="pl-k">*</span> <span class="pl-c1">1e6</span>))</td>
      </tr>
      <tr>
        <td id="L2549" class="blob-num js-line-number" data-line-number="2549"></td>
        <td id="LC2549" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_cp (J/mol K)<span class="pl-pds">&#39;</span></span>,s.get_cp()))</td>
      </tr>
      <tr>
        <td id="L2550" class="blob-num js-line-number" data-line-number="2550"></td>
        <td id="LC2550" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_cp (kJ/kg K)<span class="pl-pds">&#39;</span></span>,s.get_cp()<span class="pl-k">/</span>vrc[<span class="pl-s"><span class="pl-pds">&#39;</span>SF6<span class="pl-pds">&#39;</span></span>].mw))</td>
      </tr>
      <tr>
        <td id="L2551" class="blob-num js-line-number" data-line-number="2551"></td>
        <td id="LC2551" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_w (m/s)<span class="pl-pds">&#39;</span></span>,s.get_w()))  </td>
      </tr>
      <tr>
        <td id="L2552" class="blob-num js-line-number" data-line-number="2552"></td>
        <td id="LC2552" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}{<span class="pl-k">:8.3f</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>get_jt (K/MPa)<span class="pl-pds">&#39;</span></span>,s.get_jt() <span class="pl-k">*</span><span class="pl-c1">1e6</span>))</td>
      </tr>
      <tr>
        <td id="L2553" class="blob-num js-line-number" data-line-number="2553"></td>
        <td id="LC2553" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>dv2<span class="pl-pds">&#39;</span></span>),s.dV2(), s.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">*</span> s.temperature <span class="pl-k">/</span> s.volume<span class="pl-k">**</span><span class="pl-c1">2</span> <span class="pl-k">+</span> s.dV2(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>))</td>
      </tr>
      <tr>
        <td id="L2554" class="blob-num js-line-number" data-line-number="2554"></td>
        <td id="LC2554" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>dt2<span class="pl-pds">&#39;</span></span>),s.dT2())</td>
      </tr>
      <tr>
        <td id="L2555" class="blob-num js-line-number" data-line-number="2555"></td>
        <td id="LC2555" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>dtdv<span class="pl-pds">&#39;</span></span>),s.dTdV())</td>
      </tr>
      <tr>
        <td id="L2556" class="blob-num js-line-number" data-line-number="2556"></td>
        <td id="LC2556" class="blob-code blob-code-inner js-file-line">    xa <span class="pl-k">=</span> s.dTdV(<span class="pl-v">a</span><span class="pl-k">=</span><span class="pl-s"><span class="pl-pds">&#39;</span>res<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2557" class="blob-num js-line-number" data-line-number="2557"></td>
        <td id="LC2557" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span><span class="pl-c1">{<span class="pl-k">:25s</span>}</span><span class="pl-pds">&#39;</span></span>.format(<span class="pl-s"><span class="pl-pds">&#39;</span>dtdv<span class="pl-pds">&#39;</span></span>), <span class="pl-k">-</span>s.n_molecules <span class="pl-k">*</span> cst.k <span class="pl-k">/</span> s.volume <span class="pl-k">+</span> xa[<span class="pl-c1">1</span>])</td>
      </tr>
      <tr>
        <td id="L2558" class="blob-num js-line-number" data-line-number="2558"></td>
        <td id="LC2558" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2559" class="blob-num js-line-number" data-line-number="2559"></td>
        <td id="LC2559" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>=<span class="pl-pds">&#39;</span></span><span class="pl-k">*</span><span class="pl-c1">20</span>)</td>
      </tr>
      <tr>
        <td id="L2560" class="blob-num js-line-number" data-line-number="2560"></td>
        <td id="LC2560" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>Testing get properties with SAFTgMie<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2561" class="blob-num js-line-number" data-line-number="2561"></td>
        <td id="LC2561" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(<span class="pl-s"><span class="pl-pds">&#39;</span>=<span class="pl-pds">&#39;</span></span><span class="pl-k">*</span><span class="pl-c1">20</span>)</td>
      </tr>
      <tr>
        <td id="L2562" class="blob-num js-line-number" data-line-number="2562"></td>
        <td id="LC2562" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2563" class="blob-num js-line-number" data-line-number="2563"></td>
        <td id="LC2563" class="blob-code blob-code-inner js-file-line">    nbutane <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">1</span>))</td>
      </tr>
      <tr>
        <td id="L2564" class="blob-num js-line-number" data-line-number="2564"></td>
        <td id="LC2564" class="blob-code blob-code-inner js-file-line">    npentane <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">3</span>))</td>
      </tr>
      <tr>
        <td id="L2565" class="blob-num js-line-number" data-line-number="2565"></td>
        <td id="LC2565" class="blob-code blob-code-inner js-file-line">    noctane <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">6</span>))</td>
      </tr>
      <tr>
        <td id="L2566" class="blob-num js-line-number" data-line-number="2566"></td>
        <td id="LC2566" class="blob-code blob-code-inner js-file-line">    ndecane <span class="pl-k">=</span> GMieComponent().quick_set((<span class="pl-c1">CH3</span>,<span class="pl-c1">2</span>), (<span class="pl-c1">CH2</span>,<span class="pl-c1">8</span>))</td>
      </tr>
      <tr>
        <td id="L2567" class="blob-num js-line-number" data-line-number="2567"></td>
        <td id="LC2567" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2568" class="blob-num js-line-number" data-line-number="2568"></td>
        <td id="LC2568" class="blob-code blob-code-inner js-file-line">    C4s <span class="pl-k">=</span> SAFTgMieSystem().quick_set((nbutane, <span class="pl-c1">100</span>))</td>
      </tr>
      <tr>
        <td id="L2569" class="blob-num js-line-number" data-line-number="2569"></td>
        <td id="LC2569" class="blob-code blob-code-inner js-file-line">    C5s <span class="pl-k">=</span> SAFTgMieSystem().quick_set((npentane, <span class="pl-c1">100</span>))</td>
      </tr>
      <tr>
        <td id="L2570" class="blob-num js-line-number" data-line-number="2570"></td>
        <td id="LC2570" class="blob-code blob-code-inner js-file-line">    C8s <span class="pl-k">=</span> SAFTgMieSystem().quick_set((noctane, <span class="pl-c1">100</span>))</td>
      </tr>
      <tr>
        <td id="L2571" class="blob-num js-line-number" data-line-number="2571"></td>
        <td id="LC2571" class="blob-code blob-code-inner js-file-line">    C10s <span class="pl-k">=</span> SAFTgMieSystem().quick_set((ndecane, <span class="pl-c1">100</span>))</td>
      </tr>
      <tr>
        <td id="L2572" class="blob-num js-line-number" data-line-number="2572"></td>
        <td id="LC2572" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2573" class="blob-num js-line-number" data-line-number="2573"></td>
        <td id="LC2573" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(nbutane.cp_int(<span class="pl-c1">300</span>), nbutane.cp_t_int(<span class="pl-c1">300</span>))</td>
      </tr>
      <tr>
        <td id="L2574" class="blob-num js-line-number" data-line-number="2574"></td>
        <td id="LC2574" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(npentane.cp_int(<span class="pl-c1">300</span>), npentane.cp_t_int(<span class="pl-c1">300</span>))</td>
      </tr>
      <tr>
        <td id="L2575" class="blob-num js-line-number" data-line-number="2575"></td>
        <td id="LC2575" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2576" class="blob-num js-line-number" data-line-number="2576"></td>
        <td id="LC2576" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> C5 speed of sound</span></td>
      </tr>
      <tr>
        <td id="L2577" class="blob-num js-line-number" data-line-number="2577"></td>
        <td id="LC2577" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> (pc, tc, rhoc) = C5s.critical_point(initial_t=480., v_nd=np.logspace(-4,-1,50), get_volume=False, get_density=True, print_results=False, print_progress=False)</span></td>
      </tr>
      <tr>
        <td id="L2578" class="blob-num js-line-number" data-line-number="2578"></td>
        <td id="LC2578" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(pc, tc, 1/rhoc)</span></td>
      </tr>
      <tr>
        <td id="L2579" class="blob-num js-line-number" data-line-number="2579"></td>
        <td id="LC2579" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> trange = np.linspace(300,700,10)</span></td>
      </tr>
      <tr>
        <td id="L2580" class="blob-num js-line-number" data-line-number="2580"></td>
        <td id="LC2580" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> vget = np.zeros(10)</span></td>
      </tr>
      <tr>
        <td id="L2581" class="blob-num js-line-number" data-line-number="2581"></td>
        <td id="LC2581" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> w = np.zeros(10)</span></td>
      </tr>
      <tr>
        <td id="L2582" class="blob-num js-line-number" data-line-number="2582"></td>
        <td id="LC2582" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for i in range(len(trange)):</span></td>
      </tr>
      <tr>
        <td id="L2583" class="blob-num js-line-number" data-line-number="2583"></td>
        <td id="LC2583" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     t = trange[i]</span></td>
      </tr>
      <tr>
        <td id="L2584" class="blob-num js-line-number" data-line-number="2584"></td>
        <td id="LC2584" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     vget[i] = C5s.single_phase_v(5e6, t, print_results=False, v_crit=1/rhoc, supercritical=True)</span></td>
      </tr>
      <tr>
        <td id="L2585" class="blob-num js-line-number" data-line-number="2585"></td>
        <td id="LC2585" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     w[i] = C5s.get_w(temperature=t, molar_volume=vget[i])</span></td>
      </tr>
      <tr>
        <td id="L2586" class="blob-num js-line-number" data-line-number="2586"></td>
        <td id="LC2586" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2587" class="blob-num js-line-number" data-line-number="2587"></td>
        <td id="LC2587" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> fig, ax = plt.subplots()</span></td>
      </tr>
      <tr>
        <td id="L2588" class="blob-num js-line-number" data-line-number="2588"></td>
        <td id="LC2588" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> ax.plot(trange, w, &#39;bo&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2589" class="blob-num js-line-number" data-line-number="2589"></td>
        <td id="LC2589" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> plt.show()</span></td>
      </tr>
      <tr>
        <td id="L2590" class="blob-num js-line-number" data-line-number="2590"></td>
        <td id="LC2590" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2591" class="blob-num js-line-number" data-line-number="2591"></td>
        <td id="LC2591" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> C4 isothermal compressibility</span></td>
      </tr>
      <tr>
        <td id="L2592" class="blob-num js-line-number" data-line-number="2592"></td>
        <td id="LC2592" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> (pc, tc, rhoc) = C4s.critical_point(initial_t=380., v_nd=np.logspace(-4,-1,50), get_volume=False, get_density=True, print_results=False, print_progress=False)</span></td>
      </tr>
      <tr>
        <td id="L2593" class="blob-num js-line-number" data-line-number="2593"></td>
        <td id="LC2593" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(pc, tc, 1/rhoc)</span></td>
      </tr>
      <tr>
        <td id="L2594" class="blob-num js-line-number" data-line-number="2594"></td>
        <td id="LC2594" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> trange = np.linspace(300,700,15)</span></td>
      </tr>
      <tr>
        <td id="L2595" class="blob-num js-line-number" data-line-number="2595"></td>
        <td id="LC2595" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> vget = np.zeros(15)</span></td>
      </tr>
      <tr>
        <td id="L2596" class="blob-num js-line-number" data-line-number="2596"></td>
        <td id="LC2596" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> k = np.zeros(15)</span></td>
      </tr>
      <tr>
        <td id="L2597" class="blob-num js-line-number" data-line-number="2597"></td>
        <td id="LC2597" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for i in range(len(trange)):</span></td>
      </tr>
      <tr>
        <td id="L2598" class="blob-num js-line-number" data-line-number="2598"></td>
        <td id="LC2598" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     t = trange[i]</span></td>
      </tr>
      <tr>
        <td id="L2599" class="blob-num js-line-number" data-line-number="2599"></td>
        <td id="LC2599" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     vget[i] = C4s.single_phase_v(10e6, t, print_results=False, v_crit=1/rhoc, v_init=0.3/rhoc, supercritical=True)</span></td>
      </tr>
      <tr>
        <td id="L2600" class="blob-num js-line-number" data-line-number="2600"></td>
        <td id="LC2600" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     print(vget[i])</span></td>
      </tr>
      <tr>
        <td id="L2601" class="blob-num js-line-number" data-line-number="2601"></td>
        <td id="LC2601" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     k[i] = C4s.get_kappa(temperature=t, molar_volume=vget[i])</span></td>
      </tr>
      <tr>
        <td id="L2602" class="blob-num js-line-number" data-line-number="2602"></td>
        <td id="LC2602" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> fig, ax = plt.subplots()</span></td>
      </tr>
      <tr>
        <td id="L2603" class="blob-num js-line-number" data-line-number="2603"></td>
        <td id="LC2603" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> ax.plot(trange, k*1e9, &#39;bo&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2604" class="blob-num js-line-number" data-line-number="2604"></td>
        <td id="LC2604" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print(k * 1e9)</span></td>
      </tr>
      <tr>
        <td id="L2605" class="blob-num js-line-number" data-line-number="2605"></td>
        <td id="LC2605" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> plt.show()</span></td>
      </tr>
      <tr>
        <td id="L2606" class="blob-num js-line-number" data-line-number="2606"></td>
        <td id="LC2606" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2607" class="blob-num js-line-number" data-line-number="2607"></td>
        <td id="LC2607" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> C10 Cp</span></td>
      </tr>
      <tr>
        <td id="L2608" class="blob-num js-line-number" data-line-number="2608"></td>
        <td id="LC2608" class="blob-code blob-code-inner js-file-line">    (pc, tc, rhoc) <span class="pl-k">=</span> C10s.critical_point(<span class="pl-v">initial_t</span><span class="pl-k">=</span><span class="pl-c1">380</span>., <span class="pl-v">v_nd</span><span class="pl-k">=</span>np.logspace(<span class="pl-k">-</span><span class="pl-c1">3.5</span>,<span class="pl-k">-</span><span class="pl-c1">0.5</span>,<span class="pl-c1">50</span>), <span class="pl-v">get_volume</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">get_density</span><span class="pl-k">=</span><span class="pl-c1">True</span>, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">print_progress</span><span class="pl-k">=</span><span class="pl-c1">False</span>)</td>
      </tr>
      <tr>
        <td id="L2609" class="blob-num js-line-number" data-line-number="2609"></td>
        <td id="LC2609" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(pc, tc, <span class="pl-c1">1</span><span class="pl-k">/</span>rhoc)</td>
      </tr>
      <tr>
        <td id="L2610" class="blob-num js-line-number" data-line-number="2610"></td>
        <td id="LC2610" class="blob-code blob-code-inner js-file-line">    trange <span class="pl-k">=</span> np.linspace(<span class="pl-c1">500</span>,<span class="pl-c1">700</span>,<span class="pl-c1">15</span>)</td>
      </tr>
      <tr>
        <td id="L2611" class="blob-num js-line-number" data-line-number="2611"></td>
        <td id="LC2611" class="blob-code blob-code-inner js-file-line">    vget <span class="pl-k">=</span> np.zeros(<span class="pl-c1">15</span>)</td>
      </tr>
      <tr>
        <td id="L2612" class="blob-num js-line-number" data-line-number="2612"></td>
        <td id="LC2612" class="blob-code blob-code-inner js-file-line">    cp <span class="pl-k">=</span> np.zeros(<span class="pl-c1">15</span>)</td>
      </tr>
      <tr>
        <td id="L2613" class="blob-num js-line-number" data-line-number="2613"></td>
        <td id="LC2613" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span> i <span class="pl-k">in</span> <span class="pl-c1">range</span>(<span class="pl-c1">len</span>(trange)):</td>
      </tr>
      <tr>
        <td id="L2614" class="blob-num js-line-number" data-line-number="2614"></td>
        <td id="LC2614" class="blob-code blob-code-inner js-file-line">        t <span class="pl-k">=</span> trange[i]</td>
      </tr>
      <tr>
        <td id="L2615" class="blob-num js-line-number" data-line-number="2615"></td>
        <td id="LC2615" class="blob-code blob-code-inner js-file-line">        vget[i] <span class="pl-k">=</span> C10s.single_phase_v(<span class="pl-c1">3e6</span>, t, <span class="pl-v">print_results</span><span class="pl-k">=</span><span class="pl-c1">False</span>, <span class="pl-v">v_crit</span><span class="pl-k">=</span><span class="pl-c1">1</span><span class="pl-k">/</span>rhoc, <span class="pl-v">v_init</span><span class="pl-k">=</span><span class="pl-c1">0.3</span><span class="pl-k">/</span>rhoc, <span class="pl-v">supercritical</span><span class="pl-k">=</span><span class="pl-c1">True</span>)</td>
      </tr>
      <tr>
        <td id="L2616" class="blob-num js-line-number" data-line-number="2616"></td>
        <td id="LC2616" class="blob-code blob-code-inner js-file-line">        cp[i] <span class="pl-k">=</span> C10s.get_cp(<span class="pl-v">temperature</span><span class="pl-k">=</span>t, <span class="pl-v">molar_volume</span><span class="pl-k">=</span>vget[i])</td>
      </tr>
      <tr>
        <td id="L2617" class="blob-num js-line-number" data-line-number="2617"></td>
        <td id="LC2617" class="blob-code blob-code-inner js-file-line">    fig, ax <span class="pl-k">=</span> plt.subplots()</td>
      </tr>
      <tr>
        <td id="L2618" class="blob-num js-line-number" data-line-number="2618"></td>
        <td id="LC2618" class="blob-code blob-code-inner js-file-line">    ax.plot(trange, cp, <span class="pl-s"><span class="pl-pds">&#39;</span>bo<span class="pl-pds">&#39;</span></span>)</td>
      </tr>
      <tr>
        <td id="L2619" class="blob-num js-line-number" data-line-number="2619"></td>
        <td id="LC2619" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">print</span>(cp)</td>
      </tr>
      <tr>
        <td id="L2620" class="blob-num js-line-number" data-line-number="2620"></td>
        <td id="LC2620" class="blob-code blob-code-inner js-file-line">    plt.show()</td>
      </tr>
      <tr>
        <td id="L2621" class="blob-num js-line-number" data-line-number="2621"></td>
        <td id="LC2621" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2622" class="blob-num js-line-number" data-line-number="2622"></td>
        <td id="LC2622" class="blob-code blob-code-inner js-file-line">    <span class="pl-s"><span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2623" class="blob-num js-line-number" data-line-number="2623"></td>
        <td id="LC2623" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    comps = {}</span></td>
      </tr>
      <tr>
        <td id="L2624" class="blob-num js-line-number" data-line-number="2624"></td>
        <td id="LC2624" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    puresys = {}</span></td>
      </tr>
      <tr>
        <td id="L2625" class="blob-num js-line-number" data-line-number="2625"></td>
        <td id="LC2625" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2626" class="blob-num js-line-number" data-line-number="2626"></td>
        <td id="LC2626" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    for i in range(2,11):</span></td>
      </tr>
      <tr>
        <td id="L2627" class="blob-num js-line-number" data-line-number="2627"></td>
        <td id="LC2627" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        key = &#39;C&#39; + str(i)</span></td>
      </tr>
      <tr>
        <td id="L2628" class="blob-num js-line-number" data-line-number="2628"></td>
        <td id="LC2628" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        comps[key] = GMieComponent(i * 12.0107 + (i*2+2) * 1.00784)</span></td>
      </tr>
      <tr>
        <td id="L2629" class="blob-num js-line-number" data-line-number="2629"></td>
        <td id="LC2629" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2630" class="blob-num js-line-number" data-line-number="2630"></td>
        <td id="LC2630" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        if i == 2:</span></td>
      </tr>
      <tr>
        <td id="L2631" class="blob-num js-line-number" data-line-number="2631"></td>
        <td id="LC2631" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            # comps[key].quick_set((CH3, 2))</span></td>
      </tr>
      <tr>
        <td id="L2632" class="blob-num js-line-number" data-line-number="2632"></td>
        <td id="LC2632" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            comps[key].quick_set((C2H6, 1))</span></td>
      </tr>
      <tr>
        <td id="L2633" class="blob-num js-line-number" data-line-number="2633"></td>
        <td id="LC2633" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        else:</span></td>
      </tr>
      <tr>
        <td id="L2634" class="blob-num js-line-number" data-line-number="2634"></td>
        <td id="LC2634" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            comps[key].quick_set((CH3, 2), (CH2, i-2))</span></td>
      </tr>
      <tr>
        <td id="L2635" class="blob-num js-line-number" data-line-number="2635"></td>
        <td id="LC2635" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        puresys[key] = SAFTgMieSystem().quick_set((comps[key], 1000))</span></td>
      </tr>
      <tr>
        <td id="L2636" class="blob-num js-line-number" data-line-number="2636"></td>
        <td id="LC2636" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2637" class="blob-num js-line-number" data-line-number="2637"></td>
        <td id="LC2637" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    print(&#39;=&#39;*20)</span></td>
      </tr>
      <tr>
        <td id="L2638" class="blob-num js-line-number" data-line-number="2638"></td>
        <td id="LC2638" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    print(&#39;Testing n-alkanes&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2639" class="blob-num js-line-number" data-line-number="2639"></td>
        <td id="LC2639" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    print(&#39;=&#39;*20)</span></td>
      </tr>
      <tr>
        <td id="L2640" class="blob-num js-line-number" data-line-number="2640"></td>
        <td id="LC2640" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2641" class="blob-num js-line-number" data-line-number="2641"></td>
        <td id="LC2641" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    pc_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2642" class="blob-num js-line-number" data-line-number="2642"></td>
        <td id="LC2642" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    tc_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2643" class="blob-num js-line-number" data-line-number="2643"></td>
        <td id="LC2643" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    rhoc_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2644" class="blob-num js-line-number" data-line-number="2644"></td>
        <td id="LC2644" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    for i in range(2,5):</span></td>
      </tr>
      <tr>
        <td id="L2645" class="blob-num js-line-number" data-line-number="2645"></td>
        <td id="LC2645" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        key = &#39;C&#39; + str(i)</span></td>
      </tr>
      <tr>
        <td id="L2646" class="blob-num js-line-number" data-line-number="2646"></td>
        <td id="LC2646" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        t_init = (i+1)**0.7 * 256.77 * 0.40772</span></td>
      </tr>
      <tr>
        <td id="L2647" class="blob-num js-line-number" data-line-number="2647"></td>
        <td id="LC2647" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        v_init =  (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200.</span></td>
      </tr>
      <tr>
        <td id="L2648" class="blob-num js-line-number" data-line-number="2648"></td>
        <td id="LC2648" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        v_range = np.logspace(math.log10(v_init/2), math.log10(v_init*2), 30)</span></td>
      </tr>
      <tr>
        <td id="L2649" class="blob-num js-line-number" data-line-number="2649"></td>
        <td id="LC2649" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        (pc, tc, rhoc) = puresys[key].critical_point(initial_t=t_init, v_nd=v_range, get_volume=False, get_density=True, print_results=False, print_progress=False)</span></td>
      </tr>
      <tr>
        <td id="L2650" class="blob-num js-line-number" data-line-number="2650"></td>
        <td id="LC2650" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        pc_data = np.append(pc_data, pc*cst.patobar)</span></td>
      </tr>
      <tr>
        <td id="L2651" class="blob-num js-line-number" data-line-number="2651"></td>
        <td id="LC2651" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        tc_data = np.append(tc_data, tc)</span></td>
      </tr>
      <tr>
        <td id="L2652" class="blob-num js-line-number" data-line-number="2652"></td>
        <td id="LC2652" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        rhoc_data = np.append(rhoc_data, rhoc)</span></td>
      </tr>
      <tr>
        <td id="L2653" class="blob-num js-line-number" data-line-number="2653"></td>
        <td id="LC2653" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        print(&#39;Critical point: pressure = {:6.3f} bar, temperature = {:6.3f} K, density = {:6.3f} mol/m3&#39;.format(pc*cst.patobar, tc, rhoc))</span></td>
      </tr>
      <tr>
        <td id="L2654" class="blob-num js-line-number" data-line-number="2654"></td>
        <td id="LC2654" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2655" class="blob-num js-line-number" data-line-number="2655"></td>
        <td id="LC2655" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    df = pd.DataFrame(np.column_stack([pc_data, tc_data, rhoc_data]))</span></td>
      </tr>
      <tr>
        <td id="L2656" class="blob-num js-line-number" data-line-number="2656"></td>
        <td id="LC2656" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    outputfile = &#39;saftgmie-alkane-crit-int.csv&#39;</span></td>
      </tr>
      <tr>
        <td id="L2657" class="blob-num js-line-number" data-line-number="2657"></td>
        <td id="LC2657" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    # df.to_csv(outputfile, index=False, header=[&#39;p_c (bar)&#39;, &#39;t_c (K)&#39;, &#39;rho_c (mol/m3)&#39;])</span></td>
      </tr>
      <tr>
        <td id="L2658" class="blob-num js-line-number" data-line-number="2658"></td>
        <td id="LC2658" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    # print(f&#39;Data generation complete. Output file: {outputfile}&#39;, &#39; &#39;*5)</span></td>
      </tr>
      <tr>
        <td id="L2659" class="blob-num js-line-number" data-line-number="2659"></td>
        <td id="LC2659" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    print(df)</span></td>
      </tr>
      <tr>
        <td id="L2660" class="blob-num js-line-number" data-line-number="2660"></td>
        <td id="LC2660" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2661" class="blob-num js-line-number" data-line-number="2661"></td>
        <td id="LC2661" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    carbon = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2662" class="blob-num js-line-number" data-line-number="2662"></td>
        <td id="LC2662" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    t_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2663" class="blob-num js-line-number" data-line-number="2663"></td>
        <td id="LC2663" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    p_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2664" class="blob-num js-line-number" data-line-number="2664"></td>
        <td id="LC2664" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    rhol = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2665" class="blob-num js-line-number" data-line-number="2665"></td>
        <td id="LC2665" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    rhog = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2666" class="blob-num js-line-number" data-line-number="2666"></td>
        <td id="LC2666" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    vl_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2667" class="blob-num js-line-number" data-line-number="2667"></td>
        <td id="LC2667" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    vv_data = np.array([])</span></td>
      </tr>
      <tr>
        <td id="L2668" class="blob-num js-line-number" data-line-number="2668"></td>
        <td id="LC2668" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    for i in range(2,5):</span></td>
      </tr>
      <tr>
        <td id="L2669" class="blob-num js-line-number" data-line-number="2669"></td>
        <td id="LC2669" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        key = &#39;C&#39; + str(i)</span></td>
      </tr>
      <tr>
        <td id="L2670" class="blob-num js-line-number" data-line-number="2670"></td>
        <td id="LC2670" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        print(&#39;VLE data for {:3s} alkane&#39;.format(key), &#39;=&#39;*10)</span></td>
      </tr>
      <tr>
        <td id="L2671" class="blob-num js-line-number" data-line-number="2671"></td>
        <td id="LC2671" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        temp_range = np.linspace(tc_data[i-2]* 0.45, tc_data[i-2] * 0.995, 30) # for 30 points</span></td>
      </tr>
      <tr>
        <td id="L2672" class="blob-num js-line-number" data-line-number="2672"></td>
        <td id="LC2672" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        for j in range(len(temp_range)):</span></td>
      </tr>
      <tr>
        <td id="L2673" class="blob-num js-line-number" data-line-number="2673"></td>
        <td id="LC2673" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            t = temp_range[j]</span></td>
      </tr>
      <tr>
        <td id="L2674" class="blob-num js-line-number" data-line-number="2674"></td>
        <td id="LC2674" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            try: </span></td>
      </tr>
      <tr>
        <td id="L2675" class="blob-num js-line-number" data-line-number="2675"></td>
        <td id="LC2675" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                if j == 0:</span></td>
      </tr>
      <tr>
        <td id="L2676" class="blob-num js-line-number" data-line-number="2676"></td>
        <td id="LC2676" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    ig = (0.25/rhoc_data[i-2],100/rhoc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2677" class="blob-num js-line-number" data-line-number="2677"></td>
        <td id="LC2677" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    # ig = (0.4 * (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200., 50 * (i * 12.0107 + (i*2+2) * 1.00784) * 0.001 / 200.)</span></td>
      </tr>
      <tr>
        <td id="L2678" class="blob-num js-line-number" data-line-number="2678"></td>
        <td id="LC2678" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                else:</span></td>
      </tr>
      <tr>
        <td id="L2679" class="blob-num js-line-number" data-line-number="2679"></td>
        <td id="LC2679" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    ig = vle</span></td>
      </tr>
      <tr>
        <td id="L2680" class="blob-num js-line-number" data-line-number="2680"></td>
        <td id="LC2680" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                pv, vle = puresys[key].vapour_pressure(t, initial_guess=ig, get_volume=True, print_results=False)</span></td>
      </tr>
      <tr>
        <td id="L2681" class="blob-num js-line-number" data-line-number="2681"></td>
        <td id="LC2681" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                if abs(vle[0] - vle[1]) &lt; 1e-6:</span></td>
      </tr>
      <tr>
        <td id="L2682" class="blob-num js-line-number" data-line-number="2682"></td>
        <td id="LC2682" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    print(f&#39;VLE points solver failed to converge at meaningful results at T={t:5.2f}, points too similar ({vle[0]:7.3e}, {vle[1]:7.3e})&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2683" class="blob-num js-line-number" data-line-number="2683"></td>
        <td id="LC2683" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    vle = ig</span></td>
      </tr>
      <tr>
        <td id="L2684" class="blob-num js-line-number" data-line-number="2684"></td>
        <td id="LC2684" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                else:</span></td>
      </tr>
      <tr>
        <td id="L2685" class="blob-num js-line-number" data-line-number="2685"></td>
        <td id="LC2685" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    carbon = np.append(carbon, i)</span></td>
      </tr>
      <tr>
        <td id="L2686" class="blob-num js-line-number" data-line-number="2686"></td>
        <td id="LC2686" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    t_data = np.append(t_data, t)</span></td>
      </tr>
      <tr>
        <td id="L2687" class="blob-num js-line-number" data-line-number="2687"></td>
        <td id="LC2687" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    p_data = np.append(p_data, pv * cst.patobar)</span></td>
      </tr>
      <tr>
        <td id="L2688" class="blob-num js-line-number" data-line-number="2688"></td>
        <td id="LC2688" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    vl_data = np.append(vl_data, min(vle))</span></td>
      </tr>
      <tr>
        <td id="L2689" class="blob-num js-line-number" data-line-number="2689"></td>
        <td id="LC2689" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    vv_data = np.append(vv_data, max(vle))</span></td>
      </tr>
      <tr>
        <td id="L2690" class="blob-num js-line-number" data-line-number="2690"></td>
        <td id="LC2690" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    rhol = np.append(rhol, 1/min(vle))</span></td>
      </tr>
      <tr>
        <td id="L2691" class="blob-num js-line-number" data-line-number="2691"></td>
        <td id="LC2691" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    rhog = np.append(rhog, 1/max(vle))</span></td>
      </tr>
      <tr>
        <td id="L2692" class="blob-num js-line-number" data-line-number="2692"></td>
        <td id="LC2692" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                    print(f&#39;Getting VLE at P = {pv*cst.patobar:5.2f} bar, T = {t:5.2f} K, v_l = {min(vle):7.3e}, v_v = {max(vle):7.3e}&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2693" class="blob-num js-line-number" data-line-number="2693"></td>
        <td id="LC2693" class="blob-code blob-code-inner js-file-line"><span class="pl-s">            except:</span></td>
      </tr>
      <tr>
        <td id="L2694" class="blob-num js-line-number" data-line-number="2694"></td>
        <td id="LC2694" class="blob-code blob-code-inner js-file-line"><span class="pl-s">                print(&#39;VLE solver failed at T={t:5.2f} due to out of range operations. Current point aborted.&#39;)</span></td>
      </tr>
      <tr>
        <td id="L2695" class="blob-num js-line-number" data-line-number="2695"></td>
        <td id="LC2695" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        carbon = np.append(carbon, i)</span></td>
      </tr>
      <tr>
        <td id="L2696" class="blob-num js-line-number" data-line-number="2696"></td>
        <td id="LC2696" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        t_data = np.append(t_data, tc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2697" class="blob-num js-line-number" data-line-number="2697"></td>
        <td id="LC2697" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        p_data = np.append(p_data, pc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2698" class="blob-num js-line-number" data-line-number="2698"></td>
        <td id="LC2698" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        vl_data = np.append(vl_data, 1/rhoc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2699" class="blob-num js-line-number" data-line-number="2699"></td>
        <td id="LC2699" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        vv_data = np.append(vv_data, 1/rhoc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2700" class="blob-num js-line-number" data-line-number="2700"></td>
        <td id="LC2700" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        rhol = np.append(rhol, rhoc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2701" class="blob-num js-line-number" data-line-number="2701"></td>
        <td id="LC2701" class="blob-code blob-code-inner js-file-line"><span class="pl-s">        rhog = np.append(rhog, rhoc_data[i-2])</span></td>
      </tr>
      <tr>
        <td id="L2702" class="blob-num js-line-number" data-line-number="2702"></td>
        <td id="LC2702" class="blob-code blob-code-inner js-file-line"><span class="pl-s"></span></td>
      </tr>
      <tr>
        <td id="L2703" class="blob-num js-line-number" data-line-number="2703"></td>
        <td id="LC2703" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    df = pd.DataFrame(np.column_stack([carbon, t_data, p_data, vl_data, vv_data, rhol, rhog]))</span></td>
      </tr>
      <tr>
        <td id="L2704" class="blob-num js-line-number" data-line-number="2704"></td>
        <td id="LC2704" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    outputfile = &#39;saftgm-gc-alkanes-vle-int.csv&#39;</span></td>
      </tr>
      <tr>
        <td id="L2705" class="blob-num js-line-number" data-line-number="2705"></td>
        <td id="LC2705" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    print(df)</span></td>
      </tr>
      <tr>
        <td id="L2706" class="blob-num js-line-number" data-line-number="2706"></td>
        <td id="LC2706" class="blob-code blob-code-inner js-file-line"><span class="pl-s">    <span class="pl-pds">&#39;&#39;&#39;</span></span></td>
      </tr>
      <tr>
        <td id="L2707" class="blob-num js-line-number" data-line-number="2707"></td>
        <td id="LC2707" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2708" class="blob-num js-line-number" data-line-number="2708"></td>
        <td id="LC2708" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> vnd = np.logspace(-4,-1, 70)</span></td>
      </tr>
      <tr>
        <td id="L2709" class="blob-num js-line-number" data-line-number="2709"></td>
        <td id="LC2709" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> for lr in [8, 10, 12, 15, 20, 36]:</span></td>
      </tr>
      <tr>
        <td id="L2710" class="blob-num js-line-number" data-line-number="2710"></td>
        <td id="LC2710" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     test = VRMieComponent(1, 10, lr, 6, 3, 200)</span></td>
      </tr>
      <tr>
        <td id="L2711" class="blob-num js-line-number" data-line-number="2711"></td>
        <td id="LC2711" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     t = SAFTVRSystem().quick_set((test,1000))</span></td>
      </tr>
      <tr>
        <td id="L2712" class="blob-num js-line-number" data-line-number="2712"></td>
        <td id="LC2712" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     t_init = (lr**-0.5)/0.2 * 200</span></td>
      </tr>
      <tr>
        <td id="L2713" class="blob-num js-line-number" data-line-number="2713"></td>
        <td id="LC2713" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     v_init = 2* 3e-10**3 * cst.Na</span></td>
      </tr>
      <tr>
        <td id="L2714" class="blob-num js-line-number" data-line-number="2714"></td>
        <td id="LC2714" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     v_range = np.logspace(math.log10(v_init/2), math.log10(v_init)+1, 30)</span></td>
      </tr>
      <tr>
        <td id="L2715" class="blob-num js-line-number" data-line-number="2715"></td>
        <td id="LC2715" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     Pc, Tc, vc = t.critical_point(initial_t=t_init, v_nd=v_range, print_progress=False, get_volume=True, print_results=False)</span></td>
      </tr>
      <tr>
        <td id="L2716" class="blob-num js-line-number" data-line-number="2716"></td>
        <td id="LC2716" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     print(&#39;{:18s}&#39;.format(&#39;T_crit*:&#39;), Tc/200)</span></td>
      </tr>
      <tr>
        <td id="L2717" class="blob-num js-line-number" data-line-number="2717"></td>
        <td id="LC2717" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span>     print(&#39;{:18s}&#39;.format(&#39;rho_crit* (mol/m3):&#39;), 1/vc * cst.Na * 1e-30 * 3**3)</span></td>
      </tr>
      <tr>
        <td id="L2718" class="blob-num js-line-number" data-line-number="2718"></td>
        <td id="LC2718" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2719" class="blob-num js-line-number" data-line-number="2719"></td>
        <td id="LC2719" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L2720" class="blob-num js-line-number" data-line-number="2720"></td>
        <td id="LC2720" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> <span class="pl-c1">__name__</span> <span class="pl-k">==</span> <span class="pl-s"><span class="pl-pds">&#39;</span>__main__<span class="pl-pds">&#39;</span></span>:</td>
      </tr>
      <tr>
        <td id="L2721" class="blob-num js-line-number" data-line-number="2721"></td>
        <td id="LC2721" class="blob-code blob-code-inner js-file-line">    main()</td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 13 16" version="1.1" width="13" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm5 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM13 7.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z"/></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2" style="width:185px">
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;">
            Copy lines
          </clipboard-copy>
        </li>
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;">
            Copy permalink
          </clipboard-copy>
        </li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/kezhengzhu/ad-saftgamma/blame/cbf0700150217ff388638e3560711e052dca7c31/saftgamma.py">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" role="menuitem" href="/kezhengzhu/ad-saftgamma/issues/new">Reference in new issue</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  

  <details class="details-reset details-overlay details-overlay-dark">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>



  </div>
</div>

    </main>
  </div>
  

  </div>

        
<div class="footer container-lg width-full p-responsive" role="contentinfo">
  <div class="position-relative d-flex flex-row-reverse flex-lg-row flex-wrap flex-lg-nowrap flex-justify-center flex-lg-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
      <li class="mr-3 mr-lg-0">&copy; 2019 <span title="0.88975s from unicorn-7975694cf8-xzx95">GitHub</span>, Inc.</li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to security, text:security" href="https://github.com/security">Security</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>
    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon d-none d-lg-block mx-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>

    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 000 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 00.01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    
    <script crossorigin="anonymous" integrity="sha512-aehfjDyZ2BA4SaTBr7sCdmtvmq3V/LTfaK/zOijZQmzOKDs+QpcgzF+WqgTYVUQdKfhcfmmSOXd5GTx9FmaKhQ==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-93f7d6c0.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-WluttDR1trXe4WsPqe26HsxR11OrJTqWZwuruKqHdudWlBb9eW1eUqnku2HcMFuks88BwSLAF6DjKhAOZfKFHg==" type="application/javascript" src="https://github.githubassets.com/assets/github-bootstrap-66c2138f.js"></script>
    
    
    
  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 000 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 00.01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>

  <div aria-live="polite" class="js-global-screen-reader-notice sr-only"></div>

  </body>
</html>

