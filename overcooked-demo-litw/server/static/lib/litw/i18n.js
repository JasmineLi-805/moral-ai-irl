/*************************************************************
 * i18n.js
 *
 * Determine the two-letter locale code to use for this study.
 *
 * Author: Trevor Croxson
 * 
 * Last Modified: February 14, 2017
 * 
 * © Copyright 2017 LabintheWild.
 * For questions about this file and permission to use
 * the code, contact us at info@labinthewild.org
 *************************************************************/


module.exports = (function() {

	var params = {

		// update this list when adding new languages to your study
		// TODO(Trevor): can we handle this in a better way? Automatically?
		availableLocales: ["en", "es"],
		COOKIE_NAME: "litw_locale",
		COOKIE_EXPIRATION: function() {
			var date = new Date();
			return date.setTime(date.getTime() + (365*24*60*60*1000)); // one year
		}(),
		set: false,

		// default to English
		defaultLocale: "en",
		locale: null
	},

	_setLocale = function(locale) {
		if (params.availableLocales.indexOf(locale) > -1) {

			// if this locale is among those supported for this study, use it
			return locale;
		}

		return params.defaultLocale;
	},

	_checkQueryString = function() {
		
		// check if a locale is set in the query string
		if (!params.set) {

			// parse the query string
			var queries = {};
		  $.each(document.location.search.substr(1).split('&'),function(c, q) {
		    var i = q.split('=');
		    if (i[0]) queries[i[0].toString()] = i[1].toString();
		  });

		  if (queries.locale) {
		  	params.locale = _setLocale(queries.locale);
		  	params.set = true;
		  }			
		}
	},

	_checkCookies = function() {
		
		// check if a locale cookie exists
		if (!params.set) {
			$.each(document.cookie.split(";"), function() {
				var cookie = this.split("=");
				if (cookie[0] === params.COOKIE_NAME) {
					params.locale = _setLocale(cookie[1]);
					params.set = true;
				}
			});
		}
	},

	_checkBrowserLang = function() {

		// attempt to get the browser's default language
	  if (!params.set) {

	  	// NOTE: use navigator.languages for Chrome, which will report the
	  	// installation language (not the browser language preference) under
	  	// navigator.language
	  	var browserLocale = (navigator.languages) ?
	  		navigator.languages[0] :
				(navigator.language || navigator.userLanguage);

	  	if (browserLocale) {

	  		// simplify a locale of the form "en-US" to "en"
	  		browserLocale = browserLocale.split("-")[0];

	  		// set cookie using the browser locale value
	  		document.cookie = params.COOKIE_NAME + "=" + browserLocale +
	  			"; expires=" + params.COOKIE_EXPIRATION + "; path=/";
	  		
	  		params.locale = _setLocale(browserLocale);
	  		params.set = true;
	  	} else {

	  		// use the default locale value
	  		params.locale = params.defaultLocale;
	  		params.set = true;
	  	}
	  }
	},

	getLocale = function() {

		// Determine the locale to used, as follows:
		// 1. if a locale query parameter is passed, use this and ignore other settings
		// 2. otherwise, if a cookie is found, use this value
		// 3. otherwise, try to get the browser language and set the cookie with this value
		// 4, otherwise, use the default locale
		_checkQueryString();
		_checkCookies();
		_checkBrowserLang();

		return params.locale;
	}

	return {
		getLocale: getLocale
	}

})();