/*************************************************************
 * litw.share.0.1.js.php
 * 
 * Contains functions for social media sharing. 
 *
 * Dependencies: jQuery
 * 
 * Author: Krzysztof Gajos
 *
 * Last Modified: September 12, 2016
 * 
 * © Copyright 2016 LabintheWild
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/

/**
  Library for inserting social media sharing buttons into our experiments.  

  See test.html in this directory for an example of how to incorporate this library in your own code.

  If you have declared the open graph properties (e.g., <meta property="og:title" content="Nutrition Test" />)
  the sharing library will read the values of those properties and there is almost nothing you need to do to configure
  the system.
  Here's how you can incorporate this code into your own test (in the <head> of your index.html):
 
 	<script src="static.labinthewild.org/litw.share.0.1.js"></script>
  <script>
    $(function() {
      // For sharing on twitter it is good to have a short version of your URL; 
      // because it is not recorded among your open graph properties, it is good
      // to record it explicitly
      LITW.share.set({"shorturl": "http://shar.es/1goMQI"});
      // the makeButtons() method takes a CSS selector for DOM element(s) where the buttons
      // should be placed.  
      share.makeButtons(".share_this");
    });
  </script>

  Later in the code, once you have the participant ID you can call:
  LITW.share.set({"participantID": participantID});

  The record of all sharing events is saved in the sharing_log database on labinthewild.org
*/

//module.exports = exports = (function( exports ) {
(function( exports ) {
  "use strict";
  
  var properties = {
    // debug level (0 = no debug messages)
    debug: 0,

    // if you set participant id, it will be recorded in the database when a sharing event is recorded
    // we assume that participant id is a positive integer
    participantID: null,

    // title of the experiment (for use with Twitter and LinkedIn)
    title: null,
    // short description of the experiment (needed by LinkedIn)
    description: null,
    // by default, set to the detected URL
    // strip out any query params (in case a prior ?locale param is present)
    url: document.URL.split("?")[0],
    // good to have a short URL for Twitter; by default set to the same value as the long URL
    // UPDATE: it looks like Twitter automatically shortens URLs so  no need to set it
    shorturl: document.URL,
    // URL of an image (needed for Pinterest)
    imageurl: null,

    // include twitter button?
    twitter: true,
    // include FB button?
    fb: true,
    // include LinkedIn button?
    linkedin: true,
    // incude Pinterest?
    pinterest: true,
    // include G+?
    gplus: true,
    // include Sina Weibo?
    sinaWeibo: false,

    lang: "en",

    // the server-side script to call to record a sharing event
    dataDestination: "http://static.labinthewild.org/share/share-data.php",

    // URLs of the social media icons
    twitterIcon: "http://static.labinthewild.org/share/images/twitter.png",
    fbIcon: "http://static.labinthewild.org/share/images/fb.png",
    linkedinIcon: "http://static.labinthewild.org/share/images/linkedin.png",
    pinterestIcon: "http://static.labinthewild.org/share/images/pinterest.png",
    gplusIcon: "http://static.labinthewild.org/share/images/gplus.png",
    sinaWeiboIcon: "http://static.labinthewild.org/share/images/weibo.png",

    // constants -- do not touch
    TWITTER: "twitter",
    FB: "fb",
    LINKEDIN: "LinkedIn",
    PINTEREST: "Pinterest",
    GPLUS: "G+",
    SINAWEIBO: "sinaWeibo"
  },

  // generate the HTML containing share buttons and then put it inside all tags referred to by the selector
  makeButtons = function(selector) {
    properties.title = properties.title || $('meta[property="og:title"]').attr('content') || "A Test on LabintheWild";

		// place the buttons in the right place on the page
		$(selector).each(function(i) {
      $(this).html(_makeButtonsHelper(i));
    });

    // if sessionFlow tracking infrastructure is in place, make sure that the buttons are being tracked
    if (typeof sessionFlow != "undefined")
      sessionFlow.installListeners();
	},

  // creates a set of buttons, each with a unique id
  _makeButtonsHelper = function(count) {
    var URItitle = encodeURIComponent(properties.title);
    var URIdescription = encodeURIComponent(properties.description || $('meta[property="og:description"]').attr('content'));
    var URIurl = encodeURIComponent(properties.url || $('meta[property="og:url"]').attr('content'));
    var URIshorturl = encodeURIComponent(properties.shorturl);
    var URIimageurl = encodeURIComponent(properties.imageurl || $('meta[property="og:image"]').attr('content'));

    var buttons = "";

    if (properties.twitter) {
      buttons += '<a class="sessionFlow" id="twitterShareButton' + count 
      + '" href="https://twitter.com/intent/tweet?text=' 
      + URItitle 
      + '%20on%20%23LabintheWild&url=' + URIurl + "?locale=" + properties.lang
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.TWITTER + '\')">'
      + '<img id="' + properties.TWITTER + count + '" src="' + properties.twitterIcon + '" width="50px" alt="share this test on Twitter" /></a>\n';
    }

    if (properties.fb) {
      buttons += '<a class="sessionFlow" id="fbShareButton' + count 
      + '" href="https://www.facebook.com/sharer/sharer.php?u=' + URIurl + "?locale=" + properties.lang
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.FB + '\')">'
      + '<img id="' + properties.FB + count + '" src="' + properties.fbIcon + '" width="50px" alt="share this test on Facebook" /></a>\n';
    }

    if (properties.linkedin) {
      buttons += '<a class="sessionFlow" id="linkedinShareButton' + count 
      + '" href="https://www.linkedin.com/shareArticle?mini=true&url=' + URIurl + "?locale=" + properties.lang
      + '&title=' + URItitle
      + '&summary=' + URIdescription
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.LINKEDIN + '\')">'
      + '<img id="' + properties.LINKEDIN + count + '" src="' + properties.linkedinIcon + '" width="50px" alt="share this test on LinkedIn" /></a>\n';
    }

    if (properties.pinterest) {
      buttons += '<a class="sessionFlow" id="pinterestShareButton' + count 
      + '" href="https://www.pinterest.com/pin/create/button/?url=' + URIurl + "?locale=" + properties.lang
      + '&media=' + URIimageurl
      + '&description=' + URIdescription
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.PINTEREST + '\')">'
      + '<img id="' + properties.PINTEREST + count + '" src="' + properties.pinterestIcon + '" width="50px" alt="share this test on Pinterest" /></a>\n';
    }

    if (properties.gplus) {
      buttons += '<a class="sessionFlow" id="gplusShareButton' + count 
      + '" href="https://plus.google.com/share?url=' + URIurl + "?locale=" + properties.lang
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.GPLUS + '\')">'
      + '<img id="' + properties.GPLUS + count + '" src="' + properties.gplusIcon + '" width="50px" alt="share this test on Google+" /></a>\n';
    }

    if (properties.sinaWeibo) {
      buttons += '<a class="sessionFlow" id="sinaWeiboShareButton' + count 
      + '" href="http://service.weibo.com/share/share.php?url=' + URIurl + "?locale=" + properties.lang
      + '&title=' + URItitle 
      + '" target="_new" onclick="LITW.share.recordShare(\'' + properties.SINAWEIBO + '\')">'
      + '<img id ="' + properties.SINAWEIBO + count + '" src="' + properties.sinaWeiboIcon + '" width="50px" alt="share this test on Sina Weibo" /></a>\n';
    }

    return buttons;
  },

  // called when somebody clicked on the link to share; remember that if this function is as an event handler,
  // "this" will refer to the DOM element that was clicked and not to the share object.
  recordShare = function(destination) {
    console.log("Sharing " + properties.title + " on " + destination + " by " + properties.participantID);

    var data = {
      destination: destination,
      title: properties.title,
      url: properties.url,
      shorturl: properties.shorturl,
      participantID: properties.participantID
    };
    var dataJSON = JSON.stringify(data);
    $.ajax({
      url: properties.dataDestination,
      type: "POST",
      data: {data: dataJSON},
      success: function( data ) {
        if (properties.debug)
          console.log("Sharing event recorded");
      },
      error: function() {
        if (properties.debug)
          console.log("Failed to transmit sharing event to server");
      }
    });

    return true;
  },

  set = function(propsToChange) {
    var props = props || {};
    for (var prop in propsToChange) {
      properties[prop] = propsToChange[prop];
    }
  },

  reset = function() {
    properties.title = null;
  }

  // public methods
  exports.share = {};
  exports.share.makeButtons = makeButtons;
  exports.share.recordShare = recordShare;
  exports.share.set = set;
  exports.share.reset = reset;

})( window.LITW = window.LITW || {} );